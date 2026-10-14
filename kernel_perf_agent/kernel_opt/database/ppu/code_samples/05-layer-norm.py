# Fused layer-norm forward (baseline). Refined from tutorials/05-layer-norm.py.
"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)




def layer_norm(x, weight, bias, eps=1e-5):
    """Forward-only fused layer-norm over the last dim. x: (..., N)."""
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused[(M,)](x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y
