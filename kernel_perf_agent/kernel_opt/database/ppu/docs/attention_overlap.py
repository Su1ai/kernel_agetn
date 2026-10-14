ATTENTION_OVERLAP = """
===================================== PPU attention: overlap activation with GEMM =================
## Problem:
QK^T and P@V are Tensor-Cell-friendly GEMMs; the activation (softmax OR ReLU) between them runs on
vector units. If all warps enter that stage together, the Tensor Cell goes idle.

## Technique (Warp-Interleave principle):
- Split a CTA into two warp groups with orthogonal work (QK by KV block, PV by output channel).
- Interleave: one group's activation stage overlaps the other group's GEMM, hiding the vector work.
- Cross-cut: within a group, use (4,2)/(2,4) warp layouts to re-read the small P more and the large
  V less (V's output width >> P's M, so fewer V re-reads wins).

## Applies to:
Attention layers whose activation is softmax OR ReLU (both sit between QK and PV). Also watch TSM
supply pressure (redundant reads), not just HBM bandwidth.
"""
