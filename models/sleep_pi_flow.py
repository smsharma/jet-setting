import sys
sys.path.append("../")
sys.path.append("../models/stribor/")

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import pytorch_lightning as pl

import stribor as st

def get_exact_model(dim, hidden_dims, latent_dim, context_dim=0, n_transforms=4, set_data=False, device='cuda'):
    
    has_latent = True if context_dim > 0 else False
    
    transforms = []
    
    for _ in range(n_transforms):
        transforms.append(st.ContinuousNormalizingFlow(dim, net=st.net.DiffeqExactTraceDeepSet(dim, hidden_dims, dim, d_h=latent_dim, latent_dim=context_dim).to(device), divergence='exact', solver='dopri5', atol=1e-4, has_latent=has_latent, set_data=set_data).to(device))
        
    model = st.Flow(st.Normal(torch.zeros(dim).to(device), torch.ones(dim).to(device)), transforms).to(device)
    
    return model

class SleepPhaseFlows(pl.LightningModule):

    def __init__(self, inference_net, n_out,
                optimizer=torch.optim.AdamW, optimizer_kwargs={"weight_decay":1e-5}, lr=5e-3, 
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

        self.flow = get_exact_model(dim=self.n_param_per_ps, hidden_dims=[128, 128, 128], latent_dim=2, n_transforms=2, context_dim=int(n_out - self.n_ps_max - 1)) 

    def forward(self, x):
        x = self.inference_net(x)
        return x
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        
        return {"optimizer": optimizer, 
                    "lr_scheduler": {
                    "scheduler": self.scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "monitor": "train_loss",
                    "frequency": 1}
                }
            
    def loss(self, z_n, z_x, z_c, z_c_tot, out):

        n_batch = out.shape[0]

        probs = out[:, :self.n_ps_max + 1]
        context = out[:, self.n_ps_max + 1:]

        # Counter loss
        n_source_log_probs = probs
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        loss_counter = cross_entropy(n_source_log_probs, z_n.long()).mean()
        
        # Combine position and flux
        y = torch.cat([z_c.unsqueeze(2), z_x], dim=2)

        mask = (torch.arange(self.n_ps_max).expand(len(z_n), self.n_ps_max).type_as(z_n) < z_n.unsqueeze(1)).float()
        mask = mask.unsqueeze(-1)

        context = context.unsqueeze(-2).repeat_interleave(self.n_ps_max, dim=-2)
        log_prob = self.flow.log_prob(y, latent=context, mask=mask)
        loss_flow = -(log_prob * mask).sum() / mask.sum()
        print(loss_counter, loss_flow)
        loss =  loss_flow + loss_counter

        return loss

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