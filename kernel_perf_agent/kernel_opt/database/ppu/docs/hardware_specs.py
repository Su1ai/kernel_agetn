HARDWARE_SPECS = """
===================================== PPU 真武 810E hardware specs ================================
## Identity
- Product name: 真武 810E (PPU-ZW810E)
- Architecture codename: Beethoven
- Compute capability: 8.0 (sm80-compatible)
- Tensor Core generation: PPU0010 (MMAv1)  -- NOT PPU0015 (MMAv2, that is M890)

## Compute unit / memory hierarchy  (measured via torch.cuda.get_device_properties)
- CU (multi-processor) count: 64
- Max threads per CU: 2048
- Warp size: 32
- Shared memory per CU (TSM): 256 KB  (max per block also 256 KB)
- Registers per CU: 131072 (128 K)
- L2 cache size: 64 MB
- CU clock: 1700 MHz (max), memory clock: 1800 MHz

## DRAM / interconnect
- Memory: 96 GB HBM2e (Samsung)
- Memory bandwidth (spec): 2765 GB/s
- Memory bandwidth (measured copy, read+write): ~2.1 TB/s
- TDP: 400 W
- Form factor / interface: OAM, PCIe 5.0  (actual link observed: Gen5 x8)
- Inter-chip interconnect: ICN, 700 GB/s

## Measured tensor-core throughput (tiled GEMM, 8192^3, acc fp32) -- microbenchmark on this card
- fp16 : ~91.7 TFLOPS
- bf16 : ~91.7 TFLOPS  (same MMA path as fp16)
- fp32/tf32 : ~45 TFLOPS
- fp8 : NOT natively supported on PPU0010:
    - e4m3 -> compile error
    - e5m2 -> ~63 TFLOPS (below fp16; i.e. emulated/upcast, not a real fp8 MMA path)
- int8 : NOT supported via tl.dot (compile error)
=> Only fp16 / bf16 / tf32 are real tensor-core compute dtypes on 810E. fp8/int8 MMA are PPU0015 (M890) features.

## Other
- ECC: enabled
- Driver 2.1.0, HGGC 13.0, SDK 2.1.0-a5f865 (from ppu-smi -q)

## Usage notes for tiling:
- "wave balance" (see that node) must target 64 CUs, not M890's 39.
- Spill budget: 128K regs/CU shared by up to 2048 threads/CU.
- TSM per CU is 256 KB; tile SMEM usage directly bounds occupancy (occ = 256KB / per-block SMEM).
"""
