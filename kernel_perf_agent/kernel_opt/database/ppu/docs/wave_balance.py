WAVE_BALANCE = """
===================================== PPU wave balancing (tail elimination) =========================
## Core finding:
For memory-bound GEMM, the dominant cost is NOT tile shape but "wave quantization" (tail effect):
when total tiles is not an integer multiple of the CU count, the last partial wave wastes parallelism.

## Evidence (M890, applies in principle to 810E):
- At integer wave counts, occupancy/tile-shape differences are tiny (<=3.3pp LLC utilization).
- At fractional waves, utilization spreads by ~16pp on average (max 33.5pp).

## Techniques:
1. Wave balance: treat BLOCK_N as a continuous knob to make total tiles == CU count * integer.
2. CU balance: persistent kernel + round-robin scheduler (next = iter * gridDim.x + blockIdx.x).
3. M-padding: waste a little compute (pad BM to cover M) to read the B stripe once instead of 3x.
4. Epilogue NoTsm: bypass TSM staging (bf16x2 direct store) to unlock fine-grained BN and odd warps.

## Note for 810E:
CU count and Tensor-Cell throughput differ from M890 (810E has 1/3.6 of the 810's tensor cells).
Keep the METHOD, re-derive the CU-count constant on the actual device.
"""
