DATA_MOVEMENT = """
===================================== PPU operand feeding (two-stage path) ======================
## Model:
HBM --(AIU async, recorded on awbar)--> TSM --(each warp explicitly loads)--> registers --> Tensor Cell.

## Key facts:
- The HBM->TSM leg is asynchronous and hideable; the TSM->register->TensorCell leg consumes warp
  issue slots and TSM read bandwidth. Making only the first leg faster does not speed up the second.
- Discrete access (gather/scatter, e.g. MoE reading token rows by index) is expensive on PPU:
  contiguous sources use AIU (one descriptor per tile); discrete sources must fall back to cp.async
  with per-thread address computation.
- Warp reuse of the same K/V is NOT merged into one read: reduce redundant reads; do not just add
  more in-flight buffers.
"""
