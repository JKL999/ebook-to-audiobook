"""
Learning rate schedulers for training.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLRSchedule(_LRScheduler):
    """Warmup cosine learning rate scheduler."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            lr_scale = self.last_epoch / self.warmup_steps
        else:
            # Cosine annealing phase
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor((self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps) * 3.14159)))
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]