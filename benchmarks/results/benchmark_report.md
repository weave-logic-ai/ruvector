# RuVector Real Benchmark Report

**Date**: 2026-03-24
**Platform**: Python 3.11.2, hnswlib 0.8.0, numpy 2.4.3, Rust 1.94.0, ruvector-core (latest main)
**Machine**: aarch64 (ARM), Linux 6.1.0-44-cloud-arm64

All results are **real measurements** — no simulation, no hardcoded values.
Recall is measured against brute-force ground truth (exact cosine similarity).

---

## random-10000 (128d, 10,000 vectors)

| Engine | QPS | Recall@10 | Memory (MB) | Build (s) | p50 (ms) | p95 (ms) |
|--------|-----|-----------|-------------|-----------|----------|----------|
| numpy brute-force (baseline) | 134.8 | 1.0000 | 4.88 | 0.003 | 3.264 | 27.540 |
| hnswlib (M=16, ef_c=128, ef_s=64) | 2568.0 | 0.7572 | 0.15* | 4.514 | 0.276 | 0.568 |
| hnswlib (M=16, ef_c=200, ef_s=200) | 1899.6 | 0.9188 | 0.15* | 6.419 | 0.470 | 0.743 |
| hnswlib (M=32, ef_c=200, ef_s=200) | 1152.6 | 0.9895 | 0.15* | 7.494 | 0.730 | 1.369 |
| **ruvector-core (M=32, ef_c=200, ef_s=200)** | **443.1** | **0.9830** | **~200** | **43.940** | **1.975** | **4.069** |

## random-100000 (128d, 100,000 vectors)

| Engine | QPS | Recall@10 | Memory (MB) | Build (s) | p50 (ms) | p95 (ms) |
|--------|-----|-----------|-------------|-----------|----------|----------|
| numpy brute-force (baseline) | 69.2 | 1.0000 | 48.83 | 0.016 | 10.202 | 35.417 |
| hnswlib (M=16, ef_c=128, ef_s=64) | 1471.6 | 0.2993 | 1.53* | 72.544 | 0.607 | 0.941 |
| hnswlib (M=16, ef_c=200, ef_s=200) | 739.2 | 0.4777 | 1.53* | 114.454 | 1.201 | 2.147 |
| hnswlib (M=32, ef_c=200, ef_s=200) | 249.5 | 0.7427 | 1.53* | 395.322 | 2.567 | 11.101 |
| **ruvector-core (M=32, ef_c=200, ef_s=200)** | **85.7** | **0.8675** | **~523** | **855.646** | **10.144** | **21.850** |

*hnswlib memory shows Python-side only; C++ index memory is not captured by tracemalloc. Real memory is higher.*

---

## Analysis

### Recall: ruvector is competitive

| Scale | hnswlib (M=32) | ruvector-core (M=32) | Delta |
|-------|----------------|---------------------|-------|
| 10K | 0.9895 | **0.9830** | -0.65% |
| 100K | 0.7427 | **0.8675** | **+16.8%** |

At 10K, ruvector is within 0.65% of hnswlib recall — essentially equivalent. At 100K, **ruvector actually has BETTER recall than hnswlib** (86.75% vs 74.27%) with the same HNSW parameters. This suggests ruvector's HNSW implementation may use a different (possibly more thorough) search strategy.

### Speed: hnswlib is significantly faster

| Scale | hnswlib QPS | ruvector QPS | Ratio |
|-------|-------------|-------------|-------|
| 10K | 1152.6 | 443.1 | hnswlib is **2.6x faster** |
| 100K | 249.5 | 85.7 | hnswlib is **2.9x faster** |

ruvector is roughly 3x slower than hnswlib for search queries. This is not surprising — hnswlib is a mature, highly-optimized C++ library with SIMD intrinsics. ruvector wraps the `hnsw_rs` Rust crate which is less optimized.

However, ruvector at 86 QPS (100K) is still **faster than brute-force** (69 QPS) and provides ~87% recall — usable for our workload.

### Build Time: ruvector is much slower

| Scale | hnswlib (s) | ruvector (s) | Ratio |
|-------|-------------|-------------|-------|
| 10K | 7.5 | 44.0 | ruvector is **5.9x slower** |
| 100K | 395.3 | 855.6 | ruvector is **2.2x slower** |

Build time is where ruvector struggles most at small scale. At 100K the gap narrows to 2.2x, but still significant. For a Cognitum Seed with 100K vectors, this means ~14 min cold-start build. Acceptable for one-time index creation; problematic for frequent rebuilds.

### Memory: ruvector uses more

ruvector-core used ~523MB RSS for 100K vectors (128d). This is ~10x what the raw vectors require (100K × 128 × 4 bytes = 48.8MB). The overhead comes from the HNSW graph structure, metadata storage, and REDB persistence layer.

For the Cognitum Seed (100K vectors, limited RAM), this is a concern. The Orin NX 16GB has plenty of headroom, but the ESP32-S3 micro-HNSW (1K vectors) should be fine.

---

## Comparison with ruvector's Published Claims

| Metric | ruvector published | ruvector ACTUAL (100K) | hnswlib ACTUAL (100K) |
|--------|-------------------|----------------------|----------------------|
| Memory | **0.00 MB** | **523 MB** | ~1.53 MB (Python only) |
| Recall@10 | **1.0000** (hardcoded) | **0.8675** | 0.7427 |
| QPS | ~real | **85.7** | 249.5 |
| Build time | **0.00s** (hardcoded) | **855.6s** | 395.3s |
| vs competitors | **Simulated** (15x factor) | **Real** (this report) | **Real** |
| Simulated? | Yes | **No** | **No** |

---

## Verdict for Our Use Case

**ruvector HNSW is functional and has good recall, but is slower and heavier than hnswlib.**

For our production workload (~100K patterns in ruflo memory):
- **Recall 86.75%** is adequate for pattern matching / memory search (not life-critical)
- **85.7 QPS** means ~12ms per query — fine for non-latency-critical operations
- **523MB memory** is fine on the Orin NX (16GB) but would need quantization for smaller devices
- **Build time** of 14 min is acceptable for one-time index build, not for frequent rebuilds

**If we need better performance**, the options are:
1. Switch the vector backend to hnswlib (Rust bindings exist: `hnsw_rs` or FFI to C++ hnswlib)
2. Use FAISS for large-scale deployments
3. Optimize ruvector-core's HNSW implementation (SIMD, batch operations)
4. Use ruvector's quantization features to reduce memory

**For now**: ruvector works. The numbers are real, the recall is good, and our scale (100K) is within its capability. The published benchmarks were misleading, but the underlying HNSW implementation is sound.
