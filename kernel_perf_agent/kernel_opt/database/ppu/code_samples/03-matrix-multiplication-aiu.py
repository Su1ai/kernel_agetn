# PPU BF16 GEMM using tl.aiu_load (AIU async global->shared movement).
# Refined from t-head/triton-for-sail 03-matrix-multiplication-aiu.py.
# AIU constraints: 32B-aligned, BLOCK_M % 16 == 0, BLOCK_K 32B multiple, contiguous.
import torch, triton, triton.language as tl

@triton.jit
def matmul_kernel_aiu(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn, M, N, K,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_ptr_blk = tl.make_block_ptr(a_ptr, (M, K), (stride_am, stride_ak),
                                  (offs_am, offs_k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (1, 0))
    b_ptr_blk = tl.make_block_ptr(b_ptr, (K, N), (stride_bk, stride_bn),
                                  (offs_k, offs_bn), (BLOCK_SIZE_K, BLOCK_SIZE_N), (1, 0))
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.aiu_load(a_ptr_blk)
        b = tl.aiu_load(b_ptr_blk)
        acc = tl.dot(a, b, acc=acc)
        a_ptr_blk = tl.advance(a_ptr_blk, (0, BLOCK_SIZE_K))
        b_ptr_blk = tl.advance(b_ptr_blk, (BLOCK_SIZE_K, 0))
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(a, b, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    assert a.shape[1] == b.shape[0] and a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel_aiu[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4, num_stages=3)
    return c
