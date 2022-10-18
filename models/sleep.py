
import math
from itertools import permutations

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss

import pytorch_lightning as pl


class SleepPhase(pl.LightningModule):

    def __init__(self, inference_net, 
                optimizer=torch.optim.AdamW, optimizer_kwargs={"weight_decay":1e-5}, lr=3e-4, 
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_kwargs = {"patience":5},
                n_ps_max=5):
        super().__init__()
        
        self.save_hyperparameters("n_ps_max", "lr")
        
        self.variational_params =  {
        "loc_mean": {"dim": 2},
        "loc_logvar": {"dim": 2},
        "flux_mean": {"dim": 1},
        "flux_logvar": {"dim": 1},
    }
        
        self.n_ps_max = n_ps_max
        self.n_out = int((n_ps_max * (n_ps_max + 1) / 2) * 6 + n_ps_max + 1)

        self.indx_mats, self.last_indx = self.get_hidden_indices()
                
        for k, v in self.indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)
            
        assert self.prob_n_source_indx.shape[0] == n_ps_max + 1

        for key, value in self.indx_mats.items():
            self.indx_mats[key] = self.indx_mats[key].to('cuda:0')
                
        
        self.inference_net = inference_net
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.lr = lr
        
    def forward(self, x):
        x = self.inference_net(x)
        return x
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        
        return {"optimizer": optimizer, 
                    "lr_scheduler": {
                    "scheduler": self.scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }
    
    def get_is_on_from_n_sources(self, n_sources, max_sources):
        assert not torch.any(torch.isnan(n_sources))
        assert torch.all(n_sources >= 0)
        assert torch.all(n_sources.le(max_sources))

        is_on_array = torch.zeros(
            *n_sources.shape,
            max_sources,
            device=n_sources.device,
            dtype=torch.float,
        )

        for i in range(max_sources):
            is_on_array[..., i] = n_sources > i

        return is_on_array
        
    def get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, cached since same for all h."""

        # Initialize matrices containing the indices for each variational param.
        indx_mats = {}
        for k, param in self.variational_params.items():
            param_dim = param["dim"]
            shape = (self.n_ps_max + 1, param_dim * self.n_ps_max)
            indx_mat = torch.full(
                shape,
                1,
                dtype=torch.long,
            )
            indx_mats[k] = indx_mat

        # Add corresponding indices to the index matrices of variational params
        # for a given n_detection.
        curr_indx = 0
        for n_detections in range(1, self.n_ps_max + 1):
            for k, param in self.variational_params.items():
                param_dim = param["dim"]
                new_indx = (param_dim * n_detections) + curr_indx
                indx_mats[k][n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                    curr_indx, new_indx
                )
                curr_indx = new_indx

        for k, v in indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)

        # Assigned indices that were not used to `prob_n_source`
        indx_mats["prob_n_source"] = torch.arange(curr_indx, self.n_out)

        return indx_mats, curr_indx

    def loss(self, z_n, z_x, z_c, out):
        
        is_on = self.get_is_on_from_n_sources(z_n, self.n_ps_max)
        
        loc_mean = torch.gather(out, 1, self.indx_mats['loc_mean'][z_n.long()]) 
        loc_logvar = torch.gather(out, 1, self.indx_mats['loc_logvar'][z_n.long()])
        
        n_batch = loc_mean.shape[0]
                
        flux_mean = torch.gather(out, 1, self.indx_mats['flux_mean'][z_n.long()])
        flux_logvar = torch.gather(out, 1, self.indx_mats['flux_logvar'][z_n.long()])

        probs = out[:,self.last_indx:]
                
        n_source_log_probs = probs.softmax(dim=1).view(n_batch, self.n_ps_max + 1)
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        counter_loss = cross_entropy(n_source_log_probs, z_n.long())
                   
        # Location loss
        lm = loc_mean.view(n_batch, self.n_ps_max, 1, 2) + (is_on.unsqueeze(-1).unsqueeze(-1) == 0) * 1e16
        ls = loc_logvar.exp().sqrt().view(n_batch, self.n_ps_max, 1, 2)
        mu = z_x.view(n_batch, 1, self.n_ps_max, 2)
                
        log_prob_comb = Normal(lm, ls).log_prob(mu).sum(dim=3)  # dim-3 summed over spatial coordinates

        n_permutations = math.factorial(self.n_ps_max)
        locs_log_probs_all_perm = torch.zeros((log_prob_comb.shape[0], n_permutations))

        for i, perm in enumerate(permutations(range(self.n_ps_max))):
            locs_log_probs_all_perm[:, i] = (log_prob_comb[:, perm].diagonal(dim1=1, dim2=2) * is_on).sum(1)

        locs_loss, indx = torch.min(-locs_log_probs_all_perm, axis=1)

        # Flux loss
        lm = flux_mean.view(n_batch, self.n_ps_max, 1) 
        ls = flux_logvar.exp().sqrt().view(n_batch, self.n_ps_max, 1) 
                
        mu = z_c.view(n_batch, 1, self.n_ps_max)

        log_prob_comb = Normal(lm, ls).log_prob(mu)

        n_permutations = math.factorial(self.n_ps_max)
        fluxes_log_probs_all_perm = torch.zeros((log_prob_comb.shape[0], n_permutations))

        for i, perm in enumerate(permutations(range(self.n_ps_max))):
            fluxes_log_probs_all_perm[:, i] = (log_prob_comb[:, perm].diagonal(dim1=1, dim2=2) * is_on).sum(1)
        
        flux_loss = -torch.gather(fluxes_log_probs_all_perm, 1, indx.unsqueeze(1)).squeeze()
                                    
        return locs_loss.mean(), flux_loss.mean(), counter_loss.mean().to('cpu'), (locs_loss + counter_loss.to('cpu') + flux_loss).mean()

    def training_step(self, batch, batch_idx):
        z_n, z_x, z_c, x = batch
        out = self(x)     
        locs_loss, flux_loss, counter_loss, loss = self.loss(z_n, z_x, z_c, out)
        self.log('train_loss', loss, on_epoch=True)
        self.log('locs_loss', locs_loss, on_epoch=True)
        self.log('flux_loss', flux_loss, on_epoch=True)
        self.log('counter_loss', counter_loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        z_n, z_x, z_c, x = batch
        out = self(x)     
        _, _, _, loss = self.loss(z_n, z_x, z_c, out)
        self.log('val_loss', loss, on_epoch=True)
        return loss