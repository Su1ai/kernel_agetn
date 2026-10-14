# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Beam-selection dedup: fingerprint first, kernel source as the fallback."""

from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)


def _entry(pid: str, code: str, time_ms: float, ptx_hash: str | None = None):
    return ProgramEntry(
        program_id=pid,
        kernel_code=code,
        metrics=ProgramMetrics(time_ms=time_ms),
        problem_id="p",
        ptx_hash=ptx_hash,
    )


def test_identical_sources_merge_even_without_fingerprint():
    """Workers that return the unchanged parent report no fingerprint.

    Five of them in one round must collapse to the single fastest entry,
    otherwise the beam fills with copies of the same kernel.
    """
    entries = [
        _entry("a", "SAME", 1.0),
        _entry("b", "SAME", 0.5),
        _entry("c", "SAME", 0.8),
    ]

    out = BeamSearchStrategy._dedup_by_ptx(entries)

    assert len(out) == 1
    assert out[0].program_id == "b"          # fastest representative wins


def test_fingerprint_still_merges_different_sources():
    """Two different sources that compile to the same code are duplicates."""
    entries = [
        _entry("a", "SOURCE_ONE", 1.0, ptx_hash="FP"),
        _entry("b", "SOURCE_TWO", 0.5, ptx_hash="FP"),
    ]

    out = BeamSearchStrategy._dedup_by_ptx(entries)

    assert len(out) == 1
    assert out[0].program_id == "b"


def test_source_fallback_adopts_a_known_fingerprint():
    """An unhashed entry whose source matches a hashed one joins that group."""
    entries = [
        _entry("hashed", "SAME", 1.0, ptx_hash="FP"),
        _entry("bare", "SAME", 0.5),
    ]

    out = BeamSearchStrategy._dedup_by_ptx(entries)

    assert len(out) == 1
    assert out[0].program_id == "bare"


def test_distinct_kernels_are_kept():
    entries = [
        _entry("a", "A", 1.0, ptx_hash="FP1"),
        _entry("b", "B", 0.5, ptx_hash="FP2"),
        _entry("c", "C", 0.7),
    ]

    out = BeamSearchStrategy._dedup_by_ptx(entries)

    assert len(out) == 3
