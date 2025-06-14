# Loss Balancing Findings and Implementation

## Summary

This document summarizes our investigation into multi-objective loss balancing for the TriangulateAI project, which renders images using differentiable triangles with three loss components: VGG perceptual loss, L1 pixel loss, and LPIPS perceptual similarity.

## Problem Discovery

### Initial Issues
1. **White/Grey Collapse**: Model outputs collapsed to uniform grey/white squares
2. **Loss Imbalance**: Perceptual loss dominated (31.88) while L1 loss was minimal (0.36)
3. **Competing Objectives**: As LPIPS decreased, PSNR/SSIM increased, showing the losses were working against each other

### Root Cause Analysis
- **Early Training**: Perceptual loss found that uniform colors minimize VGG feature differences on average
- **Weight Imbalance**: Even with manual tuning (α=0.01, β=10.0), losses competed rather than cooperated
- **Different Optimization Phases**: L1 loss important early for structure, perceptual losses important later for quality

## Solution: Geometric Mean Loss

Based on research from the multi-task learning literature, we implemented a **geometric mean loss** using the log-sum approach:

```
Total Loss = log(L1 + ε) + log(Perceptual + ε) + log(LPIPS + ε)
```

### Key Benefits
1. **No Manual Weight Tuning**: Inherently balances all objectives
2. **Automatic Focus Adjustment**: Gradients scale as 1/L, giving more weight to lagging losses
3. **Forces Joint Optimization**: Total loss only decreases if ALL components improve

### Implementation Details

#### 1. GeometricMeanLoss Class (`triangulate_ai/loss/geometric_mean.py`)
```python
class GeometricMeanLoss(nn.Module):
    def __init__(self, epsilon=1e-8, min_loss_value=1e-4, regularization_weight=0.01):
        # epsilon: Prevents log(0) errors
        # min_loss_value: Threshold for L1 regularization
        # regularization_weight: Strength of regularization
```

#### 2. L1 Regularization
To prevent vanishing gradients when losses get very small:
```python
if loss < min_loss_value:
    regularization += (min_loss_value - loss)^2 * weight
```

#### 3. Configuration (`config_geometric.yaml`)
```yaml
loss:
  type: 'geometric_mean'
  epsilon: 1e-8
  min_loss_value: 1e-4
  regularization_weight: 0.01
```

## Results
- All three losses now decrease together during training
- No more collapse to uniform colors
- Model learns both pixel-level accuracy AND perceptual quality

## Future Research Directions

### 1. Uncertainty-Based Weighting (Kendall et al., 2018)
**Concept**: Learn task-specific uncertainties that automatically determine loss weights

**Implementation approach**:
```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        # Learnable log-variance parameters
        self.log_sigma_l1 = nn.Parameter(torch.zeros(1))
        self.log_sigma_perceptual = nn.Parameter(torch.zeros(1))
        self.log_sigma_lpips = nn.Parameter(torch.zeros(1))
    
    def forward(self, losses):
        # Weight each loss by learned uncertainty
        weighted_l1 = losses['l1'] / (2 * torch.exp(2 * self.log_sigma_l1)) + self.log_sigma_l1
        # Similar for other losses
```

**Benefits**:
- Principled probabilistic foundation
- Adapts to inherent noise/difficulty of each objective
- Only adds 3 trainable parameters

### 2. GradNorm (Chen et al., 2018)
**Concept**: Dynamically adjust weights to ensure all losses train at similar rates

**Key idea**:
- Monitor gradient norms for each loss
- Increase weight if a loss is learning too slowly
- Decrease weight if a loss is learning too quickly

**Implementation sketch**:
```python
# Compute gradient norms
grad_norms = []
for loss in [l1_loss, perceptual_loss, lpips_loss]:
    grads = torch.autograd.grad(loss, shared_parameters, retain_graph=True)
    grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
    grad_norms.append(grad_norm)

# Update weights to balance gradient norms
target_norm = torch.mean(grad_norms)
for i, norm in enumerate(grad_norms):
    weight_gradient = (norm - target_norm) * learning_rate
    weights[i] -= weight_gradient
```

**Benefits**:
- Directly addresses training dynamics
- Ensures no loss gets "left behind"
- Works regardless of loss scales/units

### 3. Hybrid Approaches

#### A. Staged Training with Geometric Mean
```python
if epoch < 50:
    # Stage 1: Higher regularization to establish structure
    loss_fn.regularization_weight = 0.1
elif epoch < 150:
    # Stage 2: Reduce regularization for refinement
    loss_fn.regularization_weight = 0.01
else:
    # Stage 3: Minimal regularization for final quality
    loss_fn.regularization_weight = 0.001
```

#### B. Geometric Mean + Uncertainty Weighting
Combine the benefits of both:
- Use geometric mean as base (no manual weights)
- Add learned uncertainties for task-specific adaptation
- Best of both worlds: automatic balancing + task awareness

### 4. Advanced Ideas

#### Dynamic Epsilon
Adapt epsilon based on loss magnitudes to prevent numerical issues:
```python
epsilon = max(1e-8, min(losses) * 0.01)
```

#### Loss-Specific Architectures
Consider separate encoder branches for different objectives:
- Shared backbone
- Task-specific heads
- May reduce gradient conflicts

## Monitoring and Debugging

### Key Metrics to Track
1. **Individual Losses**: Ensure all decrease together
2. **Log Losses**: Monitor for extreme values (very negative = loss near zero)
3. **Regularization**: Should be small but non-zero
4. **Gradient Norms**: For each loss component (if implementing GradNorm)

### Warning Signs
- One log-loss going to -inf (loss approaching zero too fast)
- Regularization dominating total loss
- Sudden jumps in geometric mean

## Conclusions

The geometric mean loss successfully resolved the multi-objective optimization challenge by:
1. Eliminating manual weight tuning
2. Ensuring balanced progress on all objectives
3. Preventing collapse to trivial solutions

Future work could explore uncertainty weighting or GradNorm for even more sophisticated automatic balancing, particularly as the model scales to more complex scenes or additional loss terms.

## References

1. **Geometric Mean Loss in Multi-Task Learning**: Chennupati et al., CVPR 2019
2. **Uncertainty Weighting**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
3. **GradNorm**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
4. **Multi-Objective Optimization**: Document "Combining Multiple Losses: Additive vs Multiplicative Tradeoffs"

## Code Locations

- Loss implementations: `/triangulate_ai/loss/`
- Configurations: `/config_geometric.yaml`
- Training code: `/triangulate_ai/train.py`
- Visualization: `/visualize_geometric_loss.py`