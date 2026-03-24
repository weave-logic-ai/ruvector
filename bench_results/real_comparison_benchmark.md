# RuVector Real Benchmark Results

**Generated**: 2026-03-24
**Platform**: aarch64 Linux, Rust 1.94.0, Python 3.11.2
**Method**: All measurements are real — recall measured against brute-force ground truth, memory from RSS, no simulated competitors.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| HNSW M | 32 |
| HNSW ef_construction | 200 |
| HNSW ef_search | 200 |
| Distance metric | Cosine |
| Dimensions | 128 |
| Query count | 200 (ruvector), 1000 (hnswlib) |
| Dataset | Random uniform, deterministic seed |

---

## 10,000 Vectors (128 dimensions)

| Engine | QPS | Recall@10 | Build (s) | Latency p50 (ms) | Latency p95 (ms) |
|--------|-----|-----------|-----------|-------------------|-------------------|
| numpy brute-force (baseline) | 134.8 | 1.0000 | 0.003 | 3.264 | 27.540 |
| hnswlib (M=16, ef_c=128, ef_s=64) | 2568.0 | 0.7572 | 4.514 | 0.276 | 0.568 |
| hnswlib (M=16, ef_c=200, ef_s=200) | 1899.6 | 0.9188 | 6.419 | 0.470 | 0.743 |
| hnswlib (M=32, ef_c=200, ef_s=200) | 1152.6 | 0.9895 | 7.494 | 0.730 | 1.369 |
| **ruvector-core (M=32, ef=200)** | **443.1** | **0.9830** | **43.940** | **1.975** | **4.069** |

### Analysis (10K)

- ruvector recall (98.3%) is within 0.65% of hnswlib (98.95%) — essentially equivalent search quality
- ruvector QPS (443) is 2.6x slower than hnswlib (1153)
- ruvector build time (44s) is 5.9x slower than hnswlib (7.5s)
- All engines produce correct results (verified against brute-force ground truth)

---

## 100,000 Vectors (128 dimensions)

| Engine | QPS | Recall@10 | Build (s) | Latency p50 (ms) | Latency p95 (ms) |
|--------|-----|-----------|-----------|-------------------|-------------------|
| numpy brute-force (baseline) | 69.2 | 1.0000 | 0.016 | 10.202 | 35.417 |
| hnswlib (M=16, ef_c=128, ef_s=64) | 1471.6 | 0.2993 | 72.544 | 0.607 | 0.941 |
| hnswlib (M=16, ef_c=200, ef_s=200) | 739.2 | 0.4777 | 114.454 | 1.201 | 2.147 |
| hnswlib (M=32, ef_c=200, ef_s=200) | 249.5 | 0.7427 | 395.322 | 2.567 | 11.101 |
| **ruvector-core (M=32, ef=200)** | **85.7** | **0.8675** | **855.646** | **10.144** | **21.850** |

### Analysis (100K)

- ruvector recall (86.75%) is **higher** than hnswlib (74.27%) with identical parameters
- This suggests ruvector's HNSW implementation explores more candidates (better recall, lower QPS)
- ruvector QPS (86) is 2.9x slower than hnswlib (250) but still faster than brute-force (69)
- ruvector build time (856s) is 2.2x slower than hnswlib (395s) — gap narrows at scale
- ruvector memory: ~523MB RSS for 100K vectors (includes HNSW graph + REDB persistence overhead)

---

## Comparison with Previously Published Results

The previous benchmark results in this directory (`comparison_benchmark.md`) contained:

| Issue | Details |
|-------|---------|
| **Memory: 0.00 MB** | Memory was hardcoded to 0.0 in benchmark source. Real RSS: ~523MB for 100K vectors. |
| **Recall: 100%** | Recall was hardcoded to 1.0 without ground-truth measurement. Real recall@10: 86.75-98.3% depending on scale. |
| **Simulated competitors** | Python and brute-force baselines were simulated by multiplying ruvector's own latency. This report uses real hnswlib (C++) measurements. |
| **Build Time: 0.00s** | Build time was hardcoded to 0.0. Real build: 44-856s depending on scale. |

These issues were identified in the [benchmark audit](https://github.com/ruvnet/RuVector/issues/269) and are addressed by this report.

---

## Methodology

### ruvector-core
- Rust test binary (`tests/bench_hnsw.rs`) using ruvector-core VectorDB API
- Release build (`--release`)
- Each query measured individually with `Instant::now()` wall-clock timing
- Recall computed against brute-force cosine similarity ground truth

### hnswlib
- Python 3.11 with `hnswlib` 0.8.0 (C++ via Python bindings)
- Same dataset (generated with same PRNG seed, same dimensions)
- Same HNSW parameters (M=32, ef_construction=200, ef_search=200)
- Recall computed against numpy brute-force ground truth

### Ground Truth
- numpy brute-force: exact cosine similarity, sorted, top-k
- Used as recall reference for both hnswlib and ruvector

---

## Raw Data

Machine-readable results: [`results/competitors.json`](./results/competitors.json)
