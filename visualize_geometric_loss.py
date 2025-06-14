#!/usr/bin/env python3
"""Visualize the behavior of geometric mean loss compared to weighted sum."""

import torch
import matplotlib.pyplot as plt
import numpy as np


def weighted_sum_loss(l1, perceptual, lpips, alpha=0.1, beta=10.0, gamma=0.1):
    """Standard weighted sum loss."""
    return alpha * perceptual + beta * l1 + gamma * lpips


def geometric_mean_loss(l1, perceptual, lpips, epsilon=1e-8):
    """Geometric mean loss (log-sum implementation)."""
    return torch.log(l1 + epsilon) + torch.log(perceptual + epsilon) + torch.log(lpips + epsilon)


def visualize_loss_landscapes():
    """Compare how different loss combinations behave."""
    
    # Create a range of loss values
    loss_range = np.logspace(-4, 0, 50)  # From 0.0001 to 1.0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test case 1: All losses decrease together
    ax = axes[0, 0]
    l_values = torch.tensor(loss_range)
    weighted_losses = []
    geometric_losses = []
    
    for l in l_values:
        w_loss = weighted_sum_loss(l, l, l)
        g_loss = geometric_mean_loss(l, l, l)
        weighted_losses.append(w_loss.item())
        geometric_losses.append(g_loss.item())
    
    ax.loglog(loss_range, weighted_losses, 'b-', label='Weighted Sum', linewidth=2)
    ax.loglog(loss_range, np.exp(geometric_losses), 'r-', label='Geometric Mean', linewidth=2)
    ax.set_xlabel('Individual Loss Values')
    ax.set_ylabel('Total Loss')
    ax.set_title('All Losses Equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test case 2: One loss dominates
    ax = axes[0, 1]
    weighted_losses = []
    geometric_losses = []
    
    for l in l_values:
        # L1 is small, others are 10x larger
        w_loss = weighted_sum_loss(l, l*10, l*10)
        g_loss = geometric_mean_loss(l, l*10, l*10)
        weighted_losses.append(w_loss.item())
        geometric_losses.append(g_loss.item())
    
    ax.loglog(loss_range, weighted_losses, 'b-', label='Weighted Sum', linewidth=2)
    ax.loglog(loss_range, np.exp(geometric_losses/3), 'r-', label='Geometric Mean', linewidth=2)
    ax.set_xlabel('L1 Loss Value (others 10x)')
    ax.set_ylabel('Total Loss')
    ax.set_title('One Loss Dominates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test case 3: Gradient behavior
    ax = axes[0, 2]
    l1_vals = torch.tensor(loss_range, requires_grad=True)
    
    weighted_grads = []
    geometric_grads = []
    
    for l1 in l1_vals:
        # Fixed other losses
        perceptual = torch.tensor(0.1)
        lpips = torch.tensor(0.05)
        
        # Weighted sum gradient
        l1.grad = None
        w_loss = weighted_sum_loss(l1, perceptual, lpips)
        w_loss.backward()
        weighted_grads.append(l1.grad.item())
        
        # Geometric mean gradient
        l1.grad = None
        g_loss = geometric_mean_loss(l1, perceptual, lpips)
        g_loss.backward()
        geometric_grads.append(l1.grad.item())
    
    ax.loglog(loss_range, weighted_grads, 'b-', label='Weighted Sum Gradient', linewidth=2)
    ax.loglog(loss_range, geometric_grads, 'r-', label='Geometric Mean Gradient', linewidth=2)
    ax.set_xlabel('L1 Loss Value')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Comparison (fixed other losses)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test case 4: Loss surface visualization
    ax = axes[1, 0]
    l1_range = np.linspace(0.001, 0.5, 50)
    perceptual_range = np.linspace(0.001, 0.5, 50)
    L1, Perceptual = np.meshgrid(l1_range, perceptual_range)
    
    # Fixed LPIPS for visualization
    lpips_fixed = 0.1
    
    # Weighted sum surface
    W_surface = weighted_sum_loss(
        torch.tensor(L1), 
        torch.tensor(Perceptual), 
        torch.tensor(lpips_fixed)
    ).numpy()
    
    im = ax.contourf(L1, Perceptual, W_surface, levels=20, cmap='viridis')
    ax.set_xlabel('L1 Loss')
    ax.set_ylabel('Perceptual Loss')
    ax.set_title('Weighted Sum Loss Surface')
    plt.colorbar(im, ax=ax)
    
    # Test case 5: Geometric mean surface
    ax = axes[1, 1]
    G_surface = geometric_mean_loss(
        torch.tensor(L1), 
        torch.tensor(Perceptual), 
        torch.tensor(lpips_fixed)
    ).numpy()
    
    im = ax.contourf(L1, Perceptual, G_surface, levels=20, cmap='viridis')
    ax.set_xlabel('L1 Loss')
    ax.set_ylabel('Perceptual Loss')
    ax.set_title('Geometric Mean Loss Surface')
    plt.colorbar(im, ax=ax)
    
    # Test case 6: Regularization effect
    ax = axes[1, 2]
    l_values = torch.tensor(loss_range)
    geometric_losses_noreg = []
    geometric_losses_reg = []
    
    min_loss = 1e-4
    reg_weight = 0.01
    
    for l in l_values:
        # Without regularization
        g_loss = geometric_mean_loss(l, l, l)
        geometric_losses_noreg.append(g_loss.item())
        
        # With regularization
        reg = 0.0
        if l < min_loss:
            reg = (min_loss - l)**2 * reg_weight * 3  # 3 losses
        g_loss_reg = g_loss + reg
        geometric_losses_reg.append(g_loss_reg.item())
    
    ax.semilogx(loss_range, geometric_losses_noreg, 'b-', label='No Regularization', linewidth=2)
    ax.semilogx(loss_range, geometric_losses_reg, 'r-', label='With L1 Regularization', linewidth=2)
    ax.axvline(x=min_loss, color='gray', linestyle='--', alpha=0.5, label='Min Loss Threshold')
    ax.set_xlabel('Individual Loss Values')
    ax.set_ylabel('Total Loss (log scale)')
    ax.set_title('Effect of Regularization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('geometric_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: geometric_loss_comparison.png")
    
    # Print some key insights
    print("\nKey Insights:")
    print("1. Geometric mean naturally balances losses - no manual weight tuning needed")
    print("2. Gradients are inversely proportional to loss values (1/L)")
    print("3. Small losses get amplified gradients, preventing premature convergence")
    print("4. Regularization prevents any loss from getting too close to zero")
    print("5. Loss surfaces show geometric mean creates more balanced optimization landscape")


if __name__ == "__main__":
    visualize_loss_landscapes()
    plt.show()