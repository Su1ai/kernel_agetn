TILING = """
===================================== PPU GEMM tiling rules (真武 810/810E) ======================
## Memory model (why tile shape matters):
AsyncCopy_TOTAL = MNK * (1/BLOCK_M + 1/BLOCK_N)
SharedMem_LOAD  = MNK * (warpN/BLOCK_N + warpM/BLOCK_M)
BLOCK_K does NOT change global/shared load volume.

## When the kernel spills registers, tune in this order:
1. First reduce BLOCK_K        (affects only pipeline/staging; minimal memory impact).
2. Then increase num_warps     (no change to global->shared; only shared-memory load).
3. Last reduce BLOCK_M / BLOCK_N (increases BOTH AsyncCopy and SharedMem_Load).

## When increasing tile size: prefer increasing BLOCK_M/N.

## Detect spill:
- Command:  hg objdump --dump-resource-usage
- Spill pattern: many vmem.st / vmem.ld instructions indexed by tid.

## Note:
- Spill is not always bad: loop-exterior spill is cheap; a lightly-spilling large tile can still
  beat a smaller tile. When possible, auto-tune within a reasonable range.
"""
