#!/usr/bin/env python3
"""
Real Benchmark Suite for RuVector Audit Phase 2

Benchmarks hnswlib (C++ via Python) and numpy brute-force on standard
random datasets. Measures ACTUAL QPS, recall, memory, and build time.

Results saved to benchmarks/results/ as JSON for comparison with ruvector.
"""

import json
import os
import sys
import time
import tracemalloc
import numpy as np

# Activate venv if needed
venv_path = "/tmp/bench-env/lib/python3.11/site-packages"
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import hnswlib

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_dataset(num_vectors, dimensions, num_queries=1000, seed=42):
    """Generate random vectors and queries with ground-truth neighbors."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((num_vectors, dimensions)).astype(np.float32)
    queries = rng.standard_normal((num_queries, dimensions)).astype(np.float32)

    # Compute ground-truth: brute-force exact nearest neighbors
    print(f"  Computing ground truth ({num_queries} queries × {num_vectors} vectors)...")
    gt_start = time.perf_counter()
    ground_truth = []
    for q in queries:
        # Cosine similarity = dot product of normalized vectors
        norms_data = np.linalg.norm(data, axis=1, keepdims=True)
        norms_data = np.where(norms_data == 0, 1, norms_data)
        normalized = data / norms_data
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
            norm_q = 1
        normalized_q = q / norm_q
        sims = normalized @ normalized_q
        # Top-100 nearest neighbors
        top_k = min(100, num_vectors)
        indices = np.argsort(-sims)[:top_k]
        ground_truth.append(indices.tolist())
    gt_time = time.perf_counter() - gt_start
    print(f"  Ground truth computed in {gt_time:.2f}s")

    return data, queries, ground_truth


def benchmark_brute_force(data, queries, ground_truth, dimensions):
    """Benchmark numpy brute-force cosine search."""
    print("\n=== Numpy Brute-Force (Baseline) ===")
    num_vectors = len(data)
    num_queries = len(queries)

    # Normalize data once
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized_data = data / norms

    # Build (normalize) time
    build_start = time.perf_counter()
    _ = data / norms  # re-normalize to measure
    build_time = time.perf_counter() - build_start

    # Memory
    tracemalloc.start()
    _ = normalized_data.copy()  # force allocation
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Query
    latencies = []
    results_at_k = {1: [], 10: [], 100: []}

    for i, q in enumerate(queries):
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
            norm_q = 1
        nq = q / norm_q

        t0 = time.perf_counter()
        sims = normalized_data @ nq
        top_100 = np.argsort(-sims)[:100]
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        gt = set(ground_truth[i][:100])
        for k in [1, 10, 100]:
            retrieved = set(top_100[:k].tolist())
            gt_k = set(ground_truth[i][:k])
            recall = len(retrieved & gt_k) / k if k <= len(gt) else len(retrieved & gt_k) / len(gt_k)
            results_at_k[k].append(recall)

    latencies_arr = np.array(latencies)
    qps = num_queries / (sum(latencies) / 1000)

    result = {
        "engine": "numpy-brute-force",
        "dataset": f"random-{num_vectors}",
        "dimensions": dimensions,
        "num_vectors": num_vectors,
        "num_queries": num_queries,
        "build_time_sec": round(build_time, 4),
        "memory_mb": round(mem_peak / 1024 / 1024, 2),
        "qps": round(qps, 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_1": round(float(np.mean(results_at_k[1])), 4),
        "recall_at_10": round(float(np.mean(results_at_k[10])), 4),
        "recall_at_100": round(float(np.mean(results_at_k[100])), 4),
        "simulated": False,
    }

    print(f"  QPS: {result['qps']}")
    print(f"  Recall@1: {result['recall_at_1']}, @10: {result['recall_at_10']}, @100: {result['recall_at_100']}")
    print(f"  Latency p50: {result['latency_p50_ms']}ms, p95: {result['latency_p95_ms']}ms")
    print(f"  Memory: {result['memory_mb']} MB")
    print(f"  Build time: {result['build_time_sec']}s")

    return result


def benchmark_hnswlib(data, queries, ground_truth, dimensions, ef_construction=200, M=16, ef_search=100):
    """Benchmark hnswlib HNSW index."""
    print(f"\n=== HNSWlib (ef_construction={ef_construction}, M={M}, ef_search={ef_search}) ===")
    num_vectors = len(data)
    num_queries = len(queries)

    # Build
    tracemalloc.start()
    build_start = time.perf_counter()
    index = hnswlib.Index(space='cosine', dim=dimensions)
    index.init_index(max_elements=num_vectors, ef_construction=ef_construction, M=M)
    index.add_items(data, np.arange(num_vectors))
    build_time = time.perf_counter() - build_start
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    index.set_ef(ef_search)

    # Query
    latencies = []
    results_at_k = {1: [], 10: [], 100: []}

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        labels, distances = index.knn_query(q.reshape(1, -1), k=100)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        retrieved_100 = set(labels[0].tolist())
        for k in [1, 10, 100]:
            retrieved = set(labels[0][:k].tolist())
            gt_k = set(ground_truth[i][:k])
            recall = len(retrieved & gt_k) / k
            results_at_k[k].append(recall)

    latencies_arr = np.array(latencies)
    qps = num_queries / (sum(latencies) / 1000)

    result = {
        "engine": f"hnswlib (M={M}, ef_c={ef_construction}, ef_s={ef_search})",
        "dataset": f"random-{num_vectors}",
        "dimensions": dimensions,
        "num_vectors": num_vectors,
        "num_queries": num_queries,
        "build_time_sec": round(build_time, 4),
        "memory_mb": round(mem_peak / 1024 / 1024, 2),
        "qps": round(qps, 1),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "recall_at_1": round(float(np.mean(results_at_k[1])), 4),
        "recall_at_10": round(float(np.mean(results_at_k[10])), 4),
        "recall_at_100": round(float(np.mean(results_at_k[100])), 4),
        "simulated": False,
    }

    print(f"  QPS: {result['qps']}")
    print(f"  Recall@1: {result['recall_at_1']}, @10: {result['recall_at_10']}, @100: {result['recall_at_100']}")
    print(f"  Latency p50: {result['latency_p50_ms']}ms, p95: {result['latency_p95_ms']}ms")
    print(f"  Memory: {result['memory_mb']} MB")
    print(f"  Build time: {result['build_time_sec']}s")

    return result


def run_dataset(num_vectors, dimensions, num_queries=1000):
    """Run all benchmarks on a single dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {num_vectors} vectors, {dimensions} dimensions, {num_queries} queries")
    print(f"{'='*60}")

    data, queries, ground_truth = generate_dataset(num_vectors, dimensions, num_queries)

    results = []

    # Brute force (baseline + ground truth validation)
    results.append(benchmark_brute_force(data, queries, ground_truth, dimensions))

    # HNSWlib with different configurations
    results.append(benchmark_hnswlib(data, queries, ground_truth, dimensions,
                                      ef_construction=128, M=16, ef_search=64))
    results.append(benchmark_hnswlib(data, queries, ground_truth, dimensions,
                                      ef_construction=200, M=16, ef_search=200))
    results.append(benchmark_hnswlib(data, queries, ground_truth, dimensions,
                                      ef_construction=200, M=32, ef_search=200))

    return results


def generate_report(all_results):
    """Generate markdown comparison report."""
    report = ["# RuVector Real Benchmark Report", ""]
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hnswlib_ver = getattr(hnswlib, '__version__', '0.8.x')
    report.append(f"**Platform**: Python {sys.version.split()[0]}, hnswlib {hnswlib_ver}, numpy {np.__version__}")
    report.append(f"**Machine**: {os.uname().machine}")
    report.append("")
    report.append("All results are **real measurements** — no simulation, no hardcoded values.")
    report.append("")

    # Group by dataset
    datasets = {}
    for r in all_results:
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    for ds, results in datasets.items():
        report.append(f"## {ds} ({results[0]['dimensions']}d, {results[0]['num_vectors']} vectors)")
        report.append("")
        report.append("| Engine | QPS | Recall@1 | Recall@10 | Recall@100 | Memory (MB) | Build (s) | p50 (ms) | p95 (ms) |")
        report.append("|--------|-----|----------|-----------|------------|-------------|-----------|----------|----------|")
        for r in results:
            report.append(f"| {r['engine']} | {r['qps']} | {r['recall_at_1']} | {r['recall_at_10']} | {r['recall_at_100']} | {r['memory_mb']} | {r['build_time_sec']} | {r['latency_p50_ms']} | {r['latency_p95_ms']} |")
        report.append("")

    report.append("---")
    report.append("")
    report.append("*ruvector results will be added when the Rust benchmark completes on the same datasets.*")

    return "\n".join(report)


if __name__ == "__main__":
    all_results = []

    # 10K vectors, 128 dimensions (small, fast)
    all_results.extend(run_dataset(10_000, 128, num_queries=1000))

    # 100K vectors, 128 dimensions (our production scale)
    all_results.extend(run_dataset(100_000, 128, num_queries=1000))

    # Save JSON results
    with open(os.path.join(RESULTS_DIR, "competitors.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Save markdown report
    report = generate_report(all_results)
    with open(os.path.join(RESULTS_DIR, "benchmark_report.md"), "w") as f:
        f.write(report)

    print(f"\n\nResults saved to {RESULTS_DIR}/")
    print("  - competitors.json (raw data)")
    print("  - benchmark_report.md (formatted report)")
