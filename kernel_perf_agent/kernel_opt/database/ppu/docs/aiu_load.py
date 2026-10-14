AIU_LOAD = """
===================================== PPU tl.aiu_load ===========================================
## What it is:
PPU Triton (>= 3.2.0) adds ``tl.aiu_load``, which moves data global->shared (TSM) through the
AIU unit in one descriptor-style instruction, instead of per-thread cp.async copies.

## Hard constraints (真武 810/810E):
- The source tensor must be 32-byte aligned.
- BLOCK_M must be a multiple of 16.
- Data along BLOCK_K must be a multiple of 32 bytes.
- The block tensor must be contiguous in both M and K directions.
- Supported dtype on 810/810E: b16 only (M890 additionally supports b8).

## Usage:
- Preferred form: tl.make_block_ptr -> tl.aiu_load(block_ptr) -> tl.advance(block_ptr, offsets).
- tl.make_tensor_descriptor (Triton 3.4+) auto-promotes to AIU load when the above constraints hold.
- Only use it for contiguous data. Discrete (gather/scatter) access must fall back to cp.async.
"""
