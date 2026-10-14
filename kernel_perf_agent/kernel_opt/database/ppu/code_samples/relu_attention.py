# ============ Causal ReLU self-attention (简化版，对标 level3/50) ============
# QK^T -> causal mask -> ReLU -> P@V.  ReLU 位于两次 GEMM 之间，适用 attention_overlap 原则。
import torch
import triton
import triton.language as tl


@triton.jit
def relu_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    T, D,
    stride_qt, stride_qd,
    stride_kt, stride_kd,
    stride_vt, stride_vd,
    stride_ot, stride_od,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    t_mask = offs_t < T
    d_mask = offs_d < D

    q = tl.load(q_ptr + offs_t[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=t_mask[:, None] & d_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for j in range(0, T, BLOCK_T):
        offs_j = j + tl.arange(0, BLOCK_T)
        j_mask = offs_j < T
        k = tl.load(k_ptr + offs_j[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=j_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        # causal: only attend to keys at or before the query position
        causal = offs_t[:, None] >= offs_j[None, :]
        s = tl.where(causal & t_mask[:, None] & j_mask[None, :], s, 0.0)
        p = tl.maximum(s, 0.0)  # ReLU activation (replaces softmax)
        v = tl.load(v_ptr + offs_j[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    mask=j_mask[:, None] & d_mask[None, :], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    o = acc.to(q.dtype)
    tl.store(o_ptr + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od,
             o, mask=t_mask[:, None] & d_mask[None, :])


def relu_attention(q, k, v):
    T, D = q.shape
    o = torch.empty_like(q)
    BLOCK_T = min(64, triton.next_power_of_2(T))
    BLOCK_D = min(64, triton.next_power_of_2(D))
    relu_attn_kernel[(triton.cdiv(T, BLOCK_T),)](
        q, k, v, o, T, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        1.0 / (D ** 0.5),
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return o
