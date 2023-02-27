# Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
# Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509

import models.stribor_lib as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MADE']

class MaskedLinear(nn.Linear):
    """ Same as Linear except has a configurable mask on the weights. """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    """
    Autoregressive transformation with a neural network, implemented using weight masking.
    "MADE: Masked Autoencoder for Distribution Estimation" (https://arxiv.org/abs/1502.03509)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size. Has to be multiple of `in_dim` such that each dimension
            has assigned output dimensions. `return_per_dim` governs how the output looks like
        activation (str, optional): Activation function from `torch.nn`. Default: 'Tanh'
        final_activation (str, optional): Last activation. Default: None
        num_masks (int, optional): Number of ordering ensembles. Default: 1
        natural_ordering (bool, optional): Whether to use natural ordering of dimensions,
            otherwise uses random permutations. Default: False
        return_per_dim: Whether to return in (..., in_dim, out_dim / in_dim) format,
            otherwise returns (..., out_dim). Default: False
    """
    def __init__(self, in_dim, hidden_dims, out_dim, context_dim=None, activation='GELU', final_activation=None,
                 num_masks=1, natural_ordering=False, reverse_ordering=False, return_per_dim=False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        self.activation = activation
        self.final_activation = final_activation
        self.return_per_dim = return_per_dim
        self.natural_ordering = natural_ordering
        self.reverse_ordering = reverse_ordering
        self.num_masks = num_masks
        assert self.out_dim % self.in_dim == 0, "out_dim must be integer multiple of in_dim"

        self.net = self._get_net()

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        
        if self.context_dim is not None:
            self.context_layer = nn.Linear(self.context_dim, self.hidden_dims[-1])

    def _get_net(self):
        net = []
        hs = [self.in_dim] + self.hidden_dims + [self.out_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.append(MaskedLinear(h0, h1))
            net.append(getattr(nn, self.activation)())
        net.pop() # pop the last activation for the output layer
        if self.final_activation is not None:
            net.append(getattr(nn, self.final_activation)())
        return nn.Sequential(*net)

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return # only a single seed, skip for efficiency

        L = len(self.hidden_dims)

        rng = np.random.RandomState()

        # sample the order of the inputs and the connectivity of all neurons
        if self.natural_ordering:
            self.m[-1] = np.arange(self.in_dim)
            if self.reverse_ordering:
                self.m[-1] = self.m[-1][::-1]
        else:
            self.m[-1] = rng.permutation(self.in_dim)

        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), max(self.in_dim-1, 1), size=self.hidden_dims[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        # handle the case where out_dim = in_dim * k, for integer k > 1
        if self.out_dim > self.in_dim:
            k = int(self.out_dim / self.in_dim)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x, context=None, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., in_dim)

        Returns:
            y (tensor): Output with shape (..., in_dim, out_dim / in_dim)
                if `return_per_dim=True`, else (..., out_dim)
        """
        original_shape = x.shape
                        
        if self.context_dim is not None:
            original_context_shape = context.shape
            context = context.view(-1, original_context_shape[-1])
            context = self.context_layer(context)

        x = x.view(-1, original_shape[-1])
        

        made_layers = [l for l in self.net.modules()]
                                                
        for i_l, l in enumerate(made_layers[1:]):
            x = l(x)
            if (i_l == 0) and (self.context_dim is not None):  # If there is a context layer
                x += context  # Add context after first MADE layer
            
        y = x

        y = y.view(*original_shape[:-1], -1, self.in_dim).transpose(-1, -2)

        return y if self.return_per_dim else y.reshape(*y.shape[:-2], -1)
