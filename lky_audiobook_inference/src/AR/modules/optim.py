"""
Optimizers for training.
"""

import torch
from torch.optim import Adam

class ScaledAdam(Adam):
    """Scaled Adam optimizer."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, scale=1.0):
        self.scale = scale
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    def step(self, closure=None):
        """Override step to apply scaling."""
        # Scale gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data *= self.scale
        
        return super().step(closure)