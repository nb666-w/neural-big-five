"""
================================================================================
💊 Social Dropout: A Personality-Aware Regularization Method
================================================================================

Implements Social Dropout as a Wasserstein regularization term that 
encourages cross-layer feature alignment, thereby increasing the network's
"Extraversion" score.

Mathematical formulation:
    L_total = L_task + λ · W₂(μ_shallow, μ_deep)

This forces shallow and deep layers to "speak the same language",
promoting cross-layer communication (higher Extraversion).

Theoretical connections:
    - CORAL (Sun & Saenko, 2016): Domain alignment via second-order stats
    - MMD (Gretton et al., 2012): Distribution matching via kernel trick
    - Social Dropout: Self-domain alignment across network layers

Author: 王唱晓
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SocialDropout(nn.Module):
    """
    Social Dropout: Cross-layer feature alignment regularization.
    
    Instead of randomly dropping neurons (standard Dropout, which forces
    introverted behavior), Social Dropout encourages neurons from different
    layers to interact by minimizing the Wasserstein distance between 
    layer-wise feature distributions.
    
    This can be applied as:
    1. A regularization loss term (recommended for research)
    2. A forward-pass modifier (experimental)
    
    Args:
        social_rate: Strength of the social regularization (λ)
        method: 'wasserstein' | 'coral' | 'mmd'
    """
    
    def __init__(self, social_rate=0.1, method='wasserstein'):
        super().__init__()
        self.social_rate = social_rate
        self.method = method
        self.layer_features = []
    
    def register_hooks(self, model, layer_names=None):
        """
        Register forward hooks on specified layers to capture features.
        
        Args:
            model: The neural network
            layer_names: List of layer names to hook. If None, auto-selects.
        """
        self.hooks = []
        self.layer_features = []
        
        def hook_fn(module, input, output):
            if not module.training:
                return
            if isinstance(output, torch.Tensor):
                # Global average pool to get per-sample feature vector
                if output.dim() == 4:  # Conv layer: (B, C, H, W)
                    feat = output.mean(dim=[2, 3])
                elif output.dim() == 3:  # Transformer: (B, T, D)
                    feat = output.mean(dim=1)
                else:  # FC layer: (B, D)
                    feat = output
                self.layer_features.append(feat)
        
        if layer_names is None:
            # Auto-select: hook into Conv2d and BatchNorm layers
            # (ReLU/GELU may be functional calls without modules)
            candidates = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear,
                                       nn.ReLU, nn.GELU, nn.SiLU, nn.LayerNorm)):
                    candidates.append((name, module))
            
            # Sample evenly to get ~6-10 hook points
            if len(candidates) > 10:
                step = max(1, len(candidates) // 8)
                candidates = candidates[::step]
            
            for name, module in candidates:
                h = module.register_forward_hook(hook_fn)
                self.hooks.append(h)
        else:
            for name, module in model.named_modules():
                if name in layer_names:
                    h = module.register_forward_hook(hook_fn)
                    self.hooks.append(h)
        
        print(f"  💊 Social Dropout: registered {len(self.hooks)} hooks "
              f"(method={self.method}, λ={self.social_rate})")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def compute_social_loss(self):
        """
        Compute the Social Dropout regularization loss.
        
        L_social = Σ distance(features_i, features_j) for selected layer pairs
        
        Returns:
            social_loss: scalar tensor
        """
        if len(self.layer_features) < 2:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0, device=self.layer_features[0].device)
        n_pairs = 0
        
        # Compare shallow vs deep layers (first quarter vs last quarter)
        n_layers = len(self.layer_features)
        shallow_idx = list(range(n_layers // 4))
        deep_idx = list(range(3 * n_layers // 4, n_layers))
        
        if not shallow_idx:
            shallow_idx = [0]
        if not deep_idx:
            deep_idx = [n_layers - 1]
        
        for i in shallow_idx:
            for j in deep_idx:
                feat_i = self.layer_features[i]
                feat_j = self.layer_features[j]
                
                # Align dimensions
                min_dim = min(feat_i.shape[-1], feat_j.shape[-1])
                feat_i = feat_i[..., :min_dim]
                feat_j = feat_j[..., :min_dim]
                
                if self.method == 'wasserstein':
                    loss += self._wasserstein_loss(feat_i, feat_j)
                elif self.method == 'coral':
                    loss += self._coral_loss(feat_i, feat_j)
                elif self.method == 'mmd':
                    loss += self._mmd_loss(feat_i, feat_j)
                
                n_pairs += 1
        
        # Clear features for next forward pass
        self.layer_features = []
        
        if n_pairs > 0:
            loss = loss / n_pairs
        
        return self.social_rate * loss
    
    def _wasserstein_loss(self, feat_a, feat_b):
        """
        Approximate Wasserstein-2 distance between feature distributions.
        
        For multivariate distributions, use sliced Wasserstein distance:
        Project to 1D, sort, compute W1.
        
        Reference: Bonneel et al. (2015) "Sliced and Radon Wasserstein 
        Barycenters of Distributions"
        """
        n_projections = 50
        dim = feat_a.shape[-1]
        
        # Random projection directions
        projections = torch.randn(dim, n_projections, device=feat_a.device)
        projections = F.normalize(projections, dim=0)
        
        # Project features
        proj_a = feat_a @ projections  # (B, n_proj)
        proj_b = feat_b @ projections  # (B, n_proj)
        
        # Sort along batch dimension
        proj_a_sorted = torch.sort(proj_a, dim=0)[0]
        proj_b_sorted = torch.sort(proj_b, dim=0)[0]
        
        # W2 distance for 1D sorted distributions
        loss = torch.mean((proj_a_sorted - proj_b_sorted) ** 2)
        
        return loss
    
    def _coral_loss(self, feat_a, feat_b):
        """
        CORAL loss: Minimize difference in second-order statistics.
        
        Reference: Sun & Saenko (2016) "Deep CORAL"
        """
        # Mean alignment
        mean_loss = torch.mean((feat_a.mean(0) - feat_b.mean(0)) ** 2)
        
        # Covariance alignment
        n = feat_a.shape[0]
        d = feat_a.shape[-1]
        
        cov_a = (feat_a - feat_a.mean(0)).T @ (feat_a - feat_a.mean(0)) / max(n - 1, 1)
        cov_b = (feat_b - feat_b.mean(0)).T @ (feat_b - feat_b.mean(0)) / max(n - 1, 1)
        
        cov_loss = torch.mean((cov_a - cov_b) ** 2) / (4 * d * d)
        
        return mean_loss + cov_loss
    
    def _mmd_loss(self, feat_a, feat_b):
        """
        Maximum Mean Discrepancy with Gaussian kernel.
        
        Reference: Gretton et al. (2012) "A Kernel Two-Sample Test"
        """
        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        xx = gaussian_kernel(feat_a, feat_a).mean()
        yy = gaussian_kernel(feat_b, feat_b).mean()
        xy = gaussian_kernel(feat_a, feat_b).mean()
        
        return xx + yy - 2 * xy


class SocialDropoutTrainer:
    """
    Convenience wrapper for training with Social Dropout regularization.
    
    Usage:
        trainer = SocialDropoutTrainer(model, social_rate=0.1)
        for x, y in dataloader:
            loss = trainer.train_step(x, y, criterion)
    """
    
    def __init__(self, model, social_rate=0.1, method='wasserstein', 
                 optimizer=None, lr=1e-3):
        self.model = model
        self.social_dropout = SocialDropout(social_rate, method)
        self.social_dropout.register_hooks(model)
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        self.history = {'task_loss': [], 'social_loss': [], 'total_loss': []}
    
    def train_step(self, x, y, criterion):
        """
        Perform one training step with Social Dropout regularization.
        
        Returns:
            dict with task_loss, social_loss, total_loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass (hooks capture features)
        pred = self.model(x)
        task_loss = criterion(pred, y)
        
        # Social Dropout regularization
        social_loss = self.social_dropout.compute_social_loss()
        
        total_loss = task_loss + social_loss
        total_loss.backward()
        self.optimizer.step()
        
        # Log
        self.history['task_loss'].append(task_loss.item())
        self.history['social_loss'].append(social_loss.item())
        self.history['total_loss'].append(total_loss.item())
        
        return {
            'task_loss': task_loss.item(),
            'social_loss': social_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def cleanup(self):
        """Remove hooks when done."""
        self.social_dropout.remove_hooks()
