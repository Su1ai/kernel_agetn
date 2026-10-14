# ============ Softmax over last dim (used between QK and PV in attention) ============
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def softmax(x):
    assert x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, y, N, BLOCK=BLOCK)
    return y
