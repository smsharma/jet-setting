import sys
sys.path.append("../")

import math
from itertools import permutations

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss

import pytorch_lightning as pl

from einops import rearrange, repeat

from models.flows import build_maf, build_nsf, build_mlp


class SleepPhaseFlows(pl.LightningModule):

    def __init__(self, inference_net, n_out,
                optimizer=torch.optim.AdamW, optimizer_kwargs={"weight_decay":1e-5}, lr=3e-4, 
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_kwargs = {"patience":5},
                n_ps_max=5):
        super().__init__()
        
        self.save_hyperparameters("n_ps_max", "lr")
        
        self.n_ps_max = n_ps_max

        self.n_param_per_ps = 3
        
        self.inference_net = inference_net
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.lr = lr

        # self.n_context_features = int(n_out / (self.n_ps_max + 1))
        self.n_context_features = int((n_out - (self.n_ps_max + 1)) / self.n_ps_max)

        self.flows = [build_maf(dim=int(self.n_param_per_ps * i + 1), context_features=self.n_context_features, num_transforms=8, hidden_features=128) for i in range(1, self.n_ps_max + 1)]

        # self.probs_mlp = build_mlp(input_dim=self.n_context_features, hidden_dim=64, output_dim=self.n_ps_max + 1, layers=2)

        self.flows = nn.ModuleList(self.flows)

        self.perms_list = [list(permutations(torch.arange(i))) for i in range(1, self.n_ps_max + 1)]
        self.n_perms = [len(perms) for perms in self.perms_list]

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
            
    def loss(self, z_n, z_x, z_c, z_c_tot, out):

        n_batch = out.shape[0]

        probs = out[:,:self.n_ps_max + 1]  # First part of output used as logits in counter loss
        x = out[:,self.n_ps_max + 1:].chunk(self.n_ps_max, dim=-1)  # The rest of the output is used as conditioning context for different flows

        # probs = self.probs_mlp(out[:,:self.n_context_features])  # Categorical probability logits used as direct output of inference network
        # x = out[:,self.n_context_features:].chunk(self.n_ps_max, dim=-1)  # The rest of the output is used as conditioning context for flow

        # Counter loss
        n_source_log_probs = probs
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        counter_loss = cross_entropy(n_source_log_probs, z_n.long())
        
        # Combine position and flux
        y = torch.cat([z_c.unsqueeze(2), z_x], dim=2)

        z_n = z_n.int()

        log_prob = 0.
        for i in range(1, self.n_ps_max + 1):

            n_elems = (z_n == i).sum()  # Number of elements in batch containing i "detectable" subhalos

            # All possible permutations of position + flux variables
            y_perm = rearrange(y[torch.where(z_n == i)[0]][:, self.perms_list[i - 1], :], 'ne np nps npps -> (ne np) (nps npps)', ne=n_elems, np=self.n_perms[i - 1], nps=i, npps=self.n_param_per_ps)

            # Repeat context variables over the permutations
            x_perm = rearrange(repeat(x[i - 1][z_n == i], 'ne ns -> ne np ns', np=self.n_perms[i - 1]), 'ne np ns -> (ne np) ns', ne=n_elems)

            # Add population parameter to each permutation (it will not affect the log_prob in isolation since it's repeated)
            y_tot_perm = repeat(z_c_tot[torch.where(z_n == i)[0]], 'ne -> ne np', np=self.n_perms[i - 1]).unsqueeze(2)
            y_tot_perm = rearrange(y_tot_perm, 'ne np n1 -> (ne np) n1')
            
            y_perm = torch.cat([y_perm, y_tot_perm], dim=1)  # Combine permuted versions of position + flux and population variables
            
            # Evaluate flow log-density
            log_probs = self.flows[i - 1].log_prob(y_perm, x_perm)

            # Take maximum log-prob across permutations 
            log_prob += (torch.max(rearrange(log_probs, '(ne np) -> ne np', ne=n_elems), dim=1)[0]).sum()
        
        return (- log_prob / n_batch) + counter_loss.mean()

    def training_step(self, batch, batch_idx):
        z_n, z_x, z_c, z_c_tot, x = batch
        out = self(x)     
        loss = self.loss(z_n, z_x, z_c, z_c_tot, out)
        self.log('train_loss', loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        z_n, z_x, z_c, z_c_tot, x = batch
        out = self(x)     
        loss = self.loss(z_n, z_x, z_c, z_c_tot, out)
        self.log('val_loss', loss, on_epoch=True)
        return loss