"""Quantitative evaluation metrics and benchmark harness."""

from evaluation.benchmark import BENCHMARK_SCENARIOS, run_benchmark_suite
from evaluation.metrics import compute_aggregate_benchmarks, evaluate_swarm_run

__all__ = [
    "evaluate_swarm_run",
    "compute_aggregate_benchmarks",
    "BENCHMARK_SCENARIOS",
    "run_benchmark_suite",
]

