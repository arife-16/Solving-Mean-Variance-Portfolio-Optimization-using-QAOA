import numpy as np
from typing import List, Union, Optional
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request, register_component


class AdamWOptimizer:
    """
    AdamW optimizer with cosine learning rate scheduler.
    Outperforms standard Adam in most tasks.
    """
    
    def __init__(self, 
                 params: List[np.ndarray],
                 learning_rate_init: Union[float, List[float]] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 0,
                 total_steps: Optional[int] = None,
                 amsgrad: bool = False):
        
        self.params = params
        self.learning_rate = learning_rate_init if isinstance(learning_rate_init, list) else [learning_rate_init] * len(params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.amsgrad = amsgrad
        
        # Momentum initialization
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        if amsgrad:
            self.v_max = [np.zeros_like(p) for p in params]
        self.step_count = 0
        
    def _get_lr(self, base_lr: float) -> float:
        """Cosine scheduler with warmup"""
        if self.step_count < self.warmup_steps:
            return base_lr * (self.step_count / self.warmup_steps)
        
        if self.total_steps is None:
            return base_lr
            
        progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def get_updates(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Compute parameter updates"""
        self.step_count += 1
        updates = []
        
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # Momentum update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i], self.v[i])
                v_hat = self.v_max[i] / (1 - self.beta2 ** self.step_count)
            else:
                v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            
            # Current learning rate
            current_lr = self._get_lr(self.learning_rate[i])
            
            # AdamW update (decoupled weight decay)
            param_update = current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            weight_decay_update = current_lr * self.weight_decay * param
            
            updates.append(param_update + weight_decay_update)
            
        return updates
    
    def get_component_stats(self) -> dict:
        """Get optimizer statistics for interconnect."""
        return {
            'step_count': self.step_count,
            'current_lr': [self._get_lr(lr) for lr in self.learning_rate],
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }


# Register component in interconnect
register_component('adamw_optimizer', AdamWOptimizer)