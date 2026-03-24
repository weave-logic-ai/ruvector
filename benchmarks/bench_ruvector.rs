/// Standalone ruvector-core HNSW benchmark
/// Run: cd crates/ruvector-core && cargo test --release bench_hnsw -- --nocapture
///
/// This runs as a test inside ruvector-core to avoid complex cross-crate build issues.

#[cfg(test)]
mod bench {
    use ruvector_core::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorDB, VectorEntry};
    use std::time::Instant;

    fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Simple deterministic PRNG (same seed = same vectors = reproducible)
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn brute_force_topk(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut sims: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(v, query)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.iter().take(k).map(|(i, _)| *i).collect()
    }

    #[test]
    fn bench_hnsw_10k() {
        let num_vectors = 10_000;
        let dimensions = 128;
        let num_queries = 100; // fewer for speed in test
        let k = 10;

        eprintln!("\n=== ruvector-core HNSW Benchmark: {}K vectors, {}d ===", num_vectors / 1000, dimensions);

        let data = generate_vectors(num_vectors, dimensions, 42);
        let queries = generate_vectors(num_queries, dimensions, 123);

        // Build index
        let opts = DbOptions {
            dimensions,
            distance_metric: DistanceMetric::Cosine,
            hnsw: HnswConfig {
                m: 32,
                ef_construction: 200,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut db = VectorDB::new(opts).expect("Failed to create VectorDB");

        let build_start = Instant::now();
        for (i, vec) in data.iter().enumerate() {
            let entry = VectorEntry {
                id: format!("v{}", i),
                vector: vec.clone(),
                metadata: None,
            };
            db.insert(entry).expect("Insert failed");
        }
        let build_time = build_start.elapsed();

        eprintln!("  Build time: {:.3}s ({} vectors)", build_time.as_secs_f64(), num_vectors);

        // Query
        let mut latencies = Vec::new();
        let mut recall_at_k = Vec::new();

        for query in &queries {
            let gt = brute_force_topk(&data, query, k);
            let gt_set: std::collections::HashSet<String> =
                gt.iter().map(|i| format!("v{}", i)).collect();

            let search = SearchQuery {
                vector: query.clone(),
                k,
                ..Default::default()
            };

            let t0 = Instant::now();
            let results = db.search(search).expect("Search failed");
            let latency = t0.elapsed();

            latencies.push(latency.as_secs_f64() * 1000.0); // ms

            let retrieved: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();
            let recall = retrieved.intersection(&gt_set).count() as f64 / k as f64;
            recall_at_k.push(recall);
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
        let qps = num_queries as f64 / (latencies.iter().sum::<f64>() / 1000.0);
        let avg_recall = recall_at_k.iter().sum::<f64>() / recall_at_k.len() as f64;

        eprintln!("  QPS: {:.1}", qps);
        eprintln!("  Recall@{}: {:.4}", k, avg_recall);
        eprintln!("  Latency p50: {:.3}ms, p95: {:.3}ms", p50, p95);

        // Basic assertions
        assert!(avg_recall > 0.5, "Recall@{} should be > 0.5, got {}", k, avg_recall);
        assert!(qps > 10.0, "QPS should be > 10, got {}", qps);
    }
}
