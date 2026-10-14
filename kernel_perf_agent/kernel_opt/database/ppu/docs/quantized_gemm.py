QUANTIZED_GEMM = """
===================================== PPU weight-only quantized GEMM (W4A16) ======================
## Context:
LLM decode is memory-bound (small batch, low arithmetic intensity); weight reads dominate.
Weight-only quantization (W4A16: 4-bit weight, BF16 activation) is standard. The GEMM path is
unchanged except for a dequant step that reconstructs BF16 weights before entering Tensor Cell.

## Key insight (Marlin):
W4A16 performance ceiling is set by whether the dequant cost can be hidden behind memory/compute
gaps, not by how fast the GEMM runs.

## Two dequant routes:
- VALU route (software, works on PPU001 = 810/810E): bit-manipulate FP4 -> BF16 in registers
  (~19-23 ALU instructions per int32), scale via two shifts. IALU is the single-issue bottleneck.
- MMA route (hardware, PPU0015/M890 only): uses the FP4 MMA instruction (W x identity = dequant),
  one TC instruction + 4 pcnvt, ~8 cycles. 810E does NOT have this instruction.

## Transferable rules (any platform):
- Do dequant in registers, zero smem round-trip.
- Move weight/scale layout rearrangement offline to quantization time; kernel does zero re-layout.
- Overlap dequant (ALU pipe) with MMA (TC pipe): they are different execution units.
- In memory-bound decode, shifting dequant work off the bottleneck pipe wins the most.
"""
