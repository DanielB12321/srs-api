"""Offline evaluation of similarity algorithms against the reference library."""

from .harness import (
    BenchmarkResult,
    LEAVE_ONE_DEPOSIT_OUT,
    LEAVE_ONE_SAMPLE_OUT,
    PROTOCOLS,
    load_signatures,
    majority_class_baseline,
    run_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "LEAVE_ONE_DEPOSIT_OUT",
    "LEAVE_ONE_SAMPLE_OUT",
    "PROTOCOLS",
    "load_signatures",
    "majority_class_baseline",
    "run_benchmark",
]
