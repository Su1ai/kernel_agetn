"""PPU (真武 810E) optimization knowledge hierarchy builder.

Reuses the generic OptNode/OptHierarchy tree from the NVIDIA database module, but
populates it with PPU-specific L1/L2/L3 content instead of the Hopper/TMA content.
The L3 code samples are the official t-head/triton-for-sail PPU examples plus a few
small helpers.
"""

from __future__ import annotations

from pathlib import Path

from kernel_perf_agent.kernel_opt.database.base import (
    OptHierarchy,
    OptNode,
    add_relation,
)
from kernel_perf_agent.kernel_opt.database.ppu.docs import (
    attention_overlap,
    aiu_load,
    data_movement,
    hardware_specs,
    quantized_gemm,
    tiling,
    wave_balance,
)


def build_ppu_hierarchy(code_samples_path: Path) -> OptHierarchy:
    """Build the PPU optimization hierarchy.

    Args:
        code_samples_path: directory containing the L3 Triton code samples
            (default: kernel_perf_agent/kernel_opt/database/ppu/code_samples).

    Returns:
        OptHierarchy with the root node attached to the PPU tree.
    """
    hierarchy = OptHierarchy()

    # ---------------- Level 1: bottleneck types ----------------
    optnode_latency = OptNode(
        level=1,
        dsl="text",
        opt_desc="""To optimize compute-bound kernels on PPU (真武 810E), reduce kernel latency by:
        - Increasing pipeline overlap between activation (softmax/ReLU) and GEMM stages.
        - Tuning tile shape (BLOCK_M/N/K) and num_warps to avoid register spill and fill Tensor Cell.
        """,
    )
    optnode_memory = OptNode(
        level=1,
        dsl="text",
        opt_desc="""To optimize memory-bound kernels on PPU (真武 810E), reduce data-movement cost by:
        - Using tl.aiu_load for contiguous global->shared movement (AIU path).
        - Avoiding discrete gather/scatter access, which must fall back to cp.async.
        - Reducing redundant shared-memory reads (TSM supply is a first-order bottleneck).
        - For decode: weight-only quantization, with dequant done in registers.
        """,
    )
    optnode_utilization = OptNode(
        level=1,
        dsl="text",
        opt_desc="""To improve utilization on PPU (真武 810E), increase Tensor Cell and TSM utilization by:
        - Wave balancing: choose tile shapes so total tiles is an integer multiple of the CU count.
        - Persistent scheduling to avoid tail effects.
        - Auto-tuning BLOCK sizes / num_warps / num_stages within a reasonable range.
        """,
    )
    optnode_hardware = OptNode(
        level=1,
        dsl="text",
        opt_desc=hardware_specs.HARDWARE_SPECS,
    )
    add_relation(
        hierarchy.root,
        [optnode_latency, optnode_memory, optnode_utilization, optnode_hardware],
    )

    # ---------------- Level 2: PPU techniques ----------------
    optnode_aiu_load = OptNode(level=2, dsl="text", opt_desc=aiu_load.AIU_LOAD)
    optnode_data_movement = OptNode(level=2, dsl="text", opt_desc=data_movement.DATA_MOVEMENT)
    optnode_tiling = OptNode(level=2, dsl="text", opt_desc=tiling.TILING)
    optnode_attention_overlap = OptNode(
        level=2, dsl="text", opt_desc=attention_overlap.ATTENTION_OVERLAP
    )
    optnode_quantized_gemm = OptNode(level=2, dsl="text", opt_desc=quantized_gemm.QUANTIZED_GEMM)
    optnode_wave_balance = OptNode(level=2, dsl="text", opt_desc=wave_balance.WAVE_BALANCE)

    add_relation(optnode_latency, [optnode_tiling, optnode_attention_overlap])
    add_relation(
        optnode_memory,
        [optnode_aiu_load, optnode_data_movement, optnode_quantized_gemm],
    )
    add_relation(
        optnode_utilization,
        [optnode_tiling, optnode_attention_overlap, optnode_wave_balance],
    )

    # ---------------- Level 3: refined PPU Triton samples ----------------
    def _sample(name: str) -> OptNode:
        return OptNode(
            level=3,
            dsl="triton",
            opt_desc=(code_samples_path / name).read_text(encoding="utf-8"),
        )

    optnode_matmul = _sample("03-matrix-multiplication.py")
    optnode_matmul_aiu = _sample("03-matrix-multiplication-aiu.py")
    optnode_fused_attn = _sample("06-fused-attention.py")
    optnode_fused_attn_aiu = _sample("06-fused-attention-aiu.py")
    optnode_softmax = _sample("softmax.py")
    optnode_relu_attention = _sample("relu_attention.py")
    optnode_vec_add = _sample("01-vector-add.py")
    optnode_fused_softmax = _sample("02-fused-softmax.py")
    optnode_dropout = _sample("04-low-memory-dropout.py")
    optnode_layernorm = _sample("05-layer-norm.py")
    optnode_grouped_gemm = _sample("08-grouped-gemm.py")
    optnode_persistent_matmul = _sample("09-persistent-matmul.py")

    add_relation(optnode_aiu_load, [optnode_matmul_aiu, optnode_fused_attn_aiu])
    add_relation(
        optnode_data_movement,
        [optnode_matmul, optnode_matmul_aiu, optnode_fused_attn_aiu,
         optnode_vec_add, optnode_dropout, optnode_layernorm],
    )
    add_relation(optnode_tiling, [optnode_matmul, optnode_matmul_aiu, optnode_grouped_gemm])
    add_relation(
        optnode_attention_overlap,
        [optnode_softmax, optnode_relu_attention, optnode_fused_attn,
         optnode_fused_attn_aiu, optnode_fused_softmax],
    )
    add_relation(optnode_quantized_gemm, [optnode_matmul])
    add_relation(optnode_wave_balance, [optnode_matmul, optnode_matmul_aiu, optnode_persistent_matmul])

    return hierarchy
