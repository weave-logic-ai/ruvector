# ADR-091: INT8 CNN Quantization — Domain-Driven Design Architecture

**Status**: Proposed
**Date**: 2026-03-12
**Authors**: RuVector Architecture Team
**Deciders**: ruv
**Technical Area**: INT8 Quantization / CNN Inference / AVX2 SIMD / Calibration / Edge Deployment
**Related**: ADR-090 (Ultra-Low-Bit QAT & Pi-Quantization), ADR-003 (SIMD Optimization Strategy), ADR-005 (WASM Runtime Integration)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-03-12 | RuVector Team | Initial proposal based on INT8 quantization design |

---

## 1. Context and Problem Statement

### 1.1 Current State

The `ruvector-cnn` crate provides CNN-based feature extraction for visual similarity search and embeddings. Current implementation uses FP32 throughout:

| Component | Path | Current State |
|-----------|------|---------------|
| Conv2d | `crates/ruvector-cnn/src/layers/conv.rs` | FP32 only |
| BatchNorm | `crates/ruvector-cnn/src/layers/batch_norm.rs` | FP32, not fused |
| MobileNetV3 | `crates/ruvector-cnn/src/models/mobilenet.rs` | FP32 inference |
| SIMD kernels | `crates/ruvector-cnn/src/simd/` | FP32 AVX2/NEON |
| Pooling | `crates/ruvector-cnn/src/layers/pooling.rs` | FP32 only |

### 1.2 Problem

1. **Performance ceiling**: FP32 AVX2 processes 8 values/cycle; INT8 can process 32 bytes/cycle — a 4x theoretical speedup untapped.

2. **Memory bandwidth**: Vision models are often memory-bound. FP32 uses 4 bytes/value; INT8 uses 1 byte — 4x reduction in memory traffic.

3. **Edge deployment**: Mobile and embedded devices benefit significantly from INT8. MobileNetV3 at INT8 fits better in cache and reduces power consumption.

4. **No calibration infrastructure**: Post-training quantization requires activation statistics. No calibration pipeline exists.

5. **BatchNorm not fused**: Separate BatchNorm passes waste memory bandwidth. Fusion into Conv2d is standard for inference.

### 1.3 Relationship to ADR-090

ADR-090 addresses ultra-low-bit (2-3 bit) quantization for LLMs with QAT. This ADR addresses INT8 quantization for CNNs with PTQ (post-training quantization):

| Aspect | ADR-090 (LLM) | ADR-091 (CNN) |
|--------|---------------|---------------|
| Target bits | 2-3 bit | 8 bit |
| Training | QAT with STE | PTQ with calibration |
| Primary use | Language models | Vision models |
| Key challenge | Reasoning preservation | Accuracy vs throughput |
| Quantization | Pi-constant, QuIP | Per-channel symmetric |

### 1.4 Strategic Goal

Deliver **2-4x inference speedup** over FP32 with **<1% top-1 accuracy degradation** for MobileNetV3 and similar architectures on:
- **Desktop/Server**: AVX2/AVX-512 acceleration
- **Mobile**: NEON INT8 acceleration
- **Browser**: WASM SIMD INT8 kernels
- **Edge**: Reduced memory footprint for embedded deployment

Target: MobileNetV3-Small inference in <5ms on M4 (vs ~15ms FP32).

---

## 2. Domain Analysis — Bounded Contexts

### 2.1 Strategic Domain Design

```
+====================================================================+
|             INT8 CNN QUANTIZATION SYSTEM (ADR-091)                  |
+====================================================================+
|                                                                      |
|  +------------------+    +-------------------+    +----------------+ |
|  | Quantization     |    | Calibration       |    | Inference      | |
|  | Core Domain      |--->| Domain            |--->| Domain         | |
|  |                  |    |                   |    |                | |
|  | - QuantParams    |    | - Statistics      |    | - INT8 Conv2d  | |
|  | - Quantize/Deq   |    | - Histograms      |    | - Fused BN     | |
|  | - Tensor Types   |    | - Calibration     |    | - INT8 ReLU    | |
|  | - Scale Compute  |    | - Methods         |    | - Requantize   | |
|  +--------+---------+    +--------+----------+    +-------+--------+ |
|           |                       |                        |         |
|           v                       v                        v         |
|  +------------------+    +-------------------+                       |
|  | SIMD Kernel      |    | Observability     |                       |
|  | Domain           |    | Domain            |                       |
|  |                  |    |                   |                       |
|  | - AVX2 INT8      |    | - Benchmarks      |                       |
|  | - NEON INT8      |    | - Accuracy Tests  |                       |
|  | - WASM SIMD      |    | - Profiling       |                       |
|  | - Scalar Fallback|    | - Quality Metrics |                       |
|  +------------------+    +-------------------+                       |
|                                                                      |
+====================================================================+
```

### 2.2 Bounded Context: Quantization Core Domain

**Responsibility**: Quantization primitives, parameter computation, tensor types.

**Aggregate Roots**:
- `QuantParams` — Quantization parameters (scale, zero_point, mode)
- `QuantizedTensor<T>` — Generic quantized tensor with metadata
- `QuantConfig` — Configuration for quantization workflow

**Value Objects**:
- `QuantMode` — Symmetric vs Asymmetric quantization
- `QuantGranularity` — Per-tensor vs Per-channel
- `ScaleComputation` — Scale/zero-point calculation logic

**Domain Events**:
- `TensorQuantized { layer, mode, granularity, mse }`
- `ScalesComputed { layer, num_channels, max_scale, min_scale }`
- `OverflowDetected { layer, value, clamped_to }`

**New files**:

```
crates/ruvector-cnn/src/quantization/
  mod.rs                    # Public API
  params.rs                 # QuantParams, QuantMode, QuantGranularity
  tensor.rs                 # QuantizedTensor<u8>, QuantizedTensor<i8>, QuantizedTensor<i32>
  config.rs                 # QuantConfig for workflow configuration
  ops.rs                    # quantize(), dequantize(), requantize()
  scale.rs                  # Scale computation algorithms
```

### 2.3 Bounded Context: Calibration Domain

**Responsibility**: Activation statistics collection, calibration methods, parameter optimization.

**Aggregate Roots**:
- `CalibrationStats` — Per-layer activation statistics
- `CalibrationEngine` — Orchestrates calibration workflow
- `Histogram` — Distribution tracking for percentile methods

**Value Objects**:
- `CalibrationMethod` — MinMax, Percentile, Entropy, MSE
- `CalibrationResult` — Per-layer quantization parameters
- `CalibrationConfig` — Calibration workflow settings

**Domain Events**:
- `CalibrationStarted { num_layers, num_samples, method }`
- `LayerCalibrated { layer, scale, zero_point, method }`
- `CalibrationComplete { total_layers, duration_ms }`

**New files**:

```
crates/ruvector-cnn/src/quantization/
  calibration/
    mod.rs                  # Public calibration API
    stats.rs                # CalibrationStats, running min/max/histogram
    histogram.rs            # Histogram for percentile computation
    methods.rs              # MinMax, Percentile, Entropy, MSE implementations
    engine.rs               # CalibrationEngine orchestrator
```

### 2.4 Bounded Context: Inference Domain

**Responsibility**: Quantized layer implementations, fused operations, inference pipeline.

**Aggregate Roots**:
- `QuantizedConv2d` — INT8 convolution with per-channel weights
- `QuantizedModel` — Full quantized model wrapper
- `FusedConvBN` — Conv2d + BatchNorm fusion

**Value Objects**:
- `QuantizedWeights` — Packed INT8 weights with scales
- `BiasI32` — Pre-computed bias in accumulator space
- `RequantizeParams` — Parameters for output requantization

**Domain Events**:
- `LayerFused { conv_layer, bn_layer, weight_change_pct }`
- `InferenceComplete { batch_size, latency_us, throughput_imgs_sec }`
- `RequantizationApplied { layer, input_scale, output_scale }`

**Integration with existing code**:

| Existing File | Integration |
|---------------|-------------|
| `layers/conv.rs` | Add `QuantizedConv2d` variant or wrapper |
| `layers/batch_norm.rs` | Add `fuse_into_conv()` method |
| `layers/activation.rs` | Add `relu_int8()`, `relu6_int8()` |
| `models/mobilenet.rs` | Add `QuantizedMobileNetV3` wrapper |

**New files**:

```
crates/ruvector-cnn/src/quantization/
  layers/
    mod.rs                  # Quantized layer exports
    conv.rs                 # QuantizedConv2d with INT8 forward
    fused.rs                # FusedConvBN implementation
    activation.rs           # Quantized ReLU, ReLU6, HardSwish
    pooling.rs              # Quantized average/max pooling
    linear.rs               # Quantized fully-connected layer
  model.rs                  # QuantizedModel wrapper
```

### 2.5 Bounded Context: SIMD Kernel Domain

**Responsibility**: Platform-specific INT8 SIMD implementations.

**Aggregate Roots**:
- `Int8Kernel` — Trait for INT8 SIMD operations
- `Avx2Int8Kernel` — AVX2 implementation
- `NeonInt8Kernel` — ARM NEON implementation
- `WasmInt8Kernel` — WASM SIMD128 implementation

**Value Objects**:
- `DotProductResult` — Accumulated i32 result from INT8 dot product
- `ConvTile` — Tiled convolution parameters for cache efficiency
- `PackedWeights` — Memory layout optimized for SIMD access

**Domain Events**:
- `KernelSelected { platform, isa, width_bits }`
- `TileProcessed { oh, ow, oc_chunk, cycles }`
- `SimdUtilization { theoretical_max, achieved, efficiency_pct }`

**Integration with existing code**:

| Existing File | Integration |
|---------------|-------------|
| `simd/mod.rs` | Add INT8 kernel dispatch |
| `simd/avx2.rs` | Add `dot_product_int8_avx2`, `conv_3x3_int8_avx2` |
| `simd/neon.rs` | Add `dot_product_int8_neon`, `conv_3x3_int8_neon` |

**New files**:

```
crates/ruvector-cnn/src/quantization/
  simd/
    mod.rs                  # INT8 kernel trait and dispatch
    avx2.rs                 # AVX2 INT8 kernels (_mm256_maddubs_epi16)
    neon.rs                 # NEON INT8 kernels (vdotq_s32 / vmull)
    wasm.rs                 # WASM SIMD128 INT8 kernels
    scalar.rs               # Scalar fallback for testing/compatibility
```

### 2.6 Bounded Context: Observability Domain

**Responsibility**: Benchmarking, accuracy testing, profiling, quality metrics.

**Aggregate Roots**:
- `QuantBenchSuite` — Criterion-based benchmark collection
- `AccuracyValidator` — FP32 vs INT8 accuracy comparison
- `QualityMonitor` — Runtime quality tracking

**Value Objects**:
- `BenchmarkResult` — Throughput, latency, memory metrics
- `AccuracyReport` — Per-layer and end-to-end accuracy metrics
- `ProfileSnapshot` — Cache, memory, cycle counts

**New files**:

```
crates/ruvector-cnn/benches/
  int8_quant_bench.rs       # Criterion benchmarks for INT8 kernels

crates/ruvector-cnn/src/quantization/
  validation/
    mod.rs                  # Validation exports
    accuracy.rs             # AccuracyValidator, MSE/cosine comparison
    quality.rs              # QualityMonitor for runtime checks
```

---

## 3. Decision: Architecture

### 3.1 Core Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Per-channel weights** | Critical for CNN accuracy; different channels have different weight distributions |
| **Asymmetric activations** | ReLU outputs are non-negative; asymmetric uses full [0, 255] range |
| **Fuse BatchNorm** | Eliminates extra memory pass; standard for inference |
| **AVX2 first** | Widest deployment; `_mm256_maddubs_epi16` is the key instruction |
| **Calibration-based PTQ** | No retraining required; fast deployment |

### 3.2 Quantization Scheme

```rust
// Symmetric quantization for weights (per-channel)
// w_q = round(w / scale), scale = max_abs(w) / 127

// Asymmetric quantization for activations (per-tensor)
// x_q = round(x / scale) + zero_point
// scale = (max - min) / 255
// zero_point = round(-min / scale)

pub enum QuantMode {
    /// w_q = round(w / scale), zero_point = 0
    Symmetric,
    /// x_q = round(x / scale) + zero_point
    Asymmetric,
}

pub enum QuantGranularity {
    /// Single scale for entire tensor
    PerTensor,
    /// Scale per output channel (for Conv2d weights)
    PerChannel,
}
```

### 3.3 INT8 Convolution Data Flow

```
                     Quantize                   INT8 Conv                  Dequantize
Input (f32) ───────────────────> Input (u8) ─────────────────> Acc (i32) ─────────────────> Output (f32)
                                     │                              │
                                     │                              │
                                     v                              v
                              scale_in, zp_in              scale_out = scale_in * scale_w
                                                           (per-channel)

INT8 Accumulator formula:
  acc[oc] = bias_q[oc] + Σ(input_q[ic] * weight_q[oc,ic])
          - zp_in * Σ(weight_q[oc,ic])  // zero-point correction

Dequantize:
  output[oc] = acc[oc] * scale_out[oc]
```

### 3.4 BatchNorm Fusion

```rust
/// Fuse BatchNorm into preceding Conv2d
///
/// Conv:  y = W * x + b
/// BN:    y' = γ * (y - μ) / σ + β
///
/// Fused: y' = W' * x + b'
/// Where: W' = W * (γ / σ)
///        b' = (b - μ) * (γ / σ) + β

pub fn fuse_conv_bn(conv: &Conv2d, bn: &BatchNorm) -> Conv2d {
    let scale = bn.gamma / (bn.var + eps).sqrt();
    let fused_weights = conv.weights * scale;  // broadcast per output channel
    let fused_bias = (conv.bias - bn.mean) * scale + bn.beta;
    Conv2d::new_with_weights(fused_weights, fused_bias)
}
```

### 3.5 Key AVX2 Instructions

| Instruction | Operation | Use Case |
|-------------|-----------|----------|
| `_mm256_maddubs_epi16` | u8×i8→i16, pairwise add | Core INT8 multiply |
| `_mm256_madd_epi16` | i16×i16→i32, pairwise add | Accumulate to i32 |
| `_mm256_max_epu8` | max(u8, u8) | Quantized ReLU |
| `_mm256_min_epu8` | min(u8, u8) | Quantized clamp |
| `_mm256_cvtepi32_ps` | i32→f32 | Dequantization |

---

## 4. Security

### 4.1 Threat Model

| Threat | Vector | Severity | Mitigation |
|--------|--------|----------|------------|
| Weight overflow | Crafted weights cause i8 overflow | High | Clamp to [-127, 127], assert bounds |
| Activation overflow | Extreme input values | Medium | Clamp quantized values, validate scale |
| Calibration poisoning | Adversarial calibration data | Medium | Validate calibration data distribution |
| Model tampering | Modified quantized weights | High | SHA-256 checksum verification |
| SIMD memory access | Out-of-bounds vector loads | Critical | Bounds checks before SIMD loops |

### 4.2 Bounds Enforcement

```rust
/// Safe quantization with bounds checking
pub fn quantize_symmetric(value: f32, scale: f32) -> i8 {
    let scaled = value / scale;
    let clamped = scaled.clamp(-127.0, 127.0);
    clamped.round() as i8
}

/// Safe SIMD loop with remainder handling
pub unsafe fn process_int8_avx2(data: &[u8], output: &mut [i32]) {
    let chunks = data.len() / 32;

    // SIMD loop (safe: chunks * 32 <= data.len())
    for i in 0..chunks {
        let ptr = data.as_ptr().add(i * 32);
        // ... AVX2 operations
    }

    // Scalar remainder (safe: explicit bounds)
    for i in (chunks * 32)..data.len() {
        output[i] = process_scalar(data[i]);
    }
}
```

### 4.3 Validation

```rust
/// Validate quantized model integrity
pub fn validate_quantized_model(model: &QuantizedModel) -> ValidationResult {
    // 1. Check all scales are positive and finite
    for layer in model.layers() {
        for &scale in layer.weight_scales() {
            assert!(scale > 0.0 && scale.is_finite());
        }
    }

    // 2. Check weight ranges
    for layer in model.layers() {
        for &w in layer.weights_q() {
            assert!(w >= -127 && w <= 127);
        }
    }

    // 3. Verify checksum
    let computed = model.compute_checksum();
    assert_eq!(computed, model.stored_checksum());

    ValidationResult::Ok
}
```

---

## 5. Benchmarking

### 5.1 Benchmark Suite

```rust
// benches/int8_quant_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_int8_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8-dot-product");

    for &size in &[256, 1024, 4096, 16384] {
        let a: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let b: Vec<i8> = (0..size).map(|i| ((i % 256) as i8).wrapping_sub(128)).collect();

        group.bench_with_input(
            BenchmarkId::new("avx2", size),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| unsafe { dot_product_int8_avx2(a, b) }),
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| dot_product_int8_scalar(a, b)),
        );
    }
    group.finish();
}

fn bench_int8_conv3x3(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8-conv3x3");

    // MobileNetV3-Small typical layer sizes
    for &(h, w, in_c, out_c) in &[
        (112, 112, 16, 16),   // Early layer
        (56, 56, 24, 24),     // Mid layer
        (28, 28, 40, 40),     // Later layer
        (14, 14, 112, 112),   // Deep layer
    ] {
        let input = vec![128u8; h * w * in_c];  // Centered activations
        let kernel = vec![0i8; out_c * in_c * 9];
        let bias = vec![0i32; out_c];
        let mut output = vec![0i32; h * w * out_c];

        group.bench_with_input(
            BenchmarkId::new("avx2", format!("{}x{}x{}->{}", h, w, in_c, out_c)),
            &(),
            |bench, _| bench.iter(|| unsafe {
                conv_3x3_int8_avx2(&input, 128, &kernel, &bias, &mut output,
                                   h, w, in_c, out_c, 1, 1)
            }),
        );
    }
    group.finish();
}

fn bench_quantize_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize-dequantize");

    for &size in &[1024, 65536, 1048576] {
        let fp32: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("quantize-u8", size),
            &fp32,
            |bench, data| bench.iter(|| quantize_asymmetric(data)),
        );
    }
    group.finish();
}

criterion_group!(
    int8_benches,
    bench_int8_dot_product,
    bench_int8_conv3x3,
    bench_quantize_dequantize,
);
criterion_main!(int8_benches);
```

### 5.2 Performance Targets

| Benchmark | Metric | Target | Baseline (FP32) |
|-----------|--------|--------|-----------------|
| INT8 dot product (4096) | Throughput | >40 GB/s | ~10 GB/s |
| INT8 3x3 conv (56×56×24) | Latency | <100 µs | ~300 µs |
| MobileNetV3-Small full | Latency | <5 ms | ~15 ms |
| MobileNetV3-Small full | Throughput | >200 img/s | ~70 img/s |
| Quantization overhead | Per-layer | <10 µs | N/A |
| Calibration (100 images) | Total time | <30 s | N/A |

### 5.3 Accuracy Targets

| Model | Metric | Target | FP32 Baseline |
|-------|--------|--------|---------------|
| MobileNetV3-Small | Top-1 Acc | >67% | 67.4% |
| MobileNetV3-Large | Top-1 Acc | >74% | 75.2% |
| Embedding cosine sim | vs FP32 | >0.995 | 1.0 |

---

## 6. Optimization

### 6.1 Memory Layout

```rust
/// Optimal weight layout for AVX2 INT8 convolution
///
/// Standard: [out_c, kh, kw, in_c] — poor cache locality
/// Optimized: [out_c/8, kh, kw, in_c, 8] — 8 output channels packed
///
/// This allows loading 8 output channel weights contiguously
/// for processing with a single AVX2 register.
pub struct PackedWeights {
    data: Vec<i8>,
    out_channels: usize,
    kernel_size: usize,
    in_channels: usize,
}

impl PackedWeights {
    pub fn from_standard(weights: &[i8], out_c: usize, ks: usize, in_c: usize) -> Self {
        let mut packed = vec![0i8; weights.len()];
        let oc_groups = (out_c + 7) / 8;

        for oc_group in 0..oc_groups {
            for kh in 0..ks {
                for kw in 0..ks {
                    for ic in 0..in_c {
                        for oc_offset in 0..8 {
                            let oc = oc_group * 8 + oc_offset;
                            if oc < out_c {
                                let src_idx = oc * ks * ks * in_c + kh * ks * in_c + kw * in_c + ic;
                                let dst_idx = (oc_group * ks * ks * in_c + kh * ks * in_c + kw * in_c) * 8 + ic * 8 + oc_offset;
                                packed[dst_idx] = weights[src_idx];
                            }
                        }
                    }
                }
            }
        }

        Self { data: packed, out_channels: out_c, kernel_size: ks, in_channels: in_c }
    }
}
```

### 6.2 Cache Tiling

```rust
/// Tiled convolution for L1/L2 cache efficiency
///
/// Process output in tiles that fit in L1 cache:
/// - Tile activations: ~32KB (L1 data cache)
/// - Tile weights: ~32KB (share across tiles)
/// - Accumulator: ~8KB (8 output channels × tile size × 4 bytes)
pub const TILE_H: usize = 8;
pub const TILE_W: usize = 8;
pub const TILE_OC: usize = 8;

pub fn conv_3x3_int8_tiled(
    input: &[u8],
    weights: &PackedWeights,
    output: &mut [i32],
    h: usize, w: usize, in_c: usize, out_c: usize,
) {
    let out_h = h;
    let out_w = w;

    // Tile over output spatial dimensions
    for oh_tile in (0..out_h).step_by(TILE_H) {
        for ow_tile in (0..out_w).step_by(TILE_W) {
            // Tile over output channels
            for oc_tile in (0..out_c).step_by(TILE_OC) {
                process_tile(
                    input, weights, output,
                    oh_tile, ow_tile, oc_tile,
                    h, w, in_c, out_c,
                );
            }
        }
    }
}
```

### 6.3 Zero-Point Optimization

```rust
/// Pre-compute zero-point correction term
///
/// For each output channel: correction = zp_input × Σ(weights)
/// This is constant per output channel and can be folded into bias.
pub fn compute_zp_correction(
    weights: &[i8],
    zero_point: i32,
    out_c: usize,
    in_c: usize,
    ks: usize,
) -> Vec<i32> {
    let mut corrections = vec![0i32; out_c];

    for oc in 0..out_c {
        let mut weight_sum = 0i32;
        for ic in 0..in_c {
            for k in 0..(ks * ks) {
                let idx = oc * in_c * ks * ks + ic * ks * ks + k;
                weight_sum += weights[idx] as i32;
            }
        }
        corrections[oc] = zero_point * weight_sum;
    }

    corrections
}

/// Fuse zero-point correction into bias
pub fn fuse_zp_into_bias(bias: &mut [i32], zp_correction: &[i32]) {
    for (b, &corr) in bias.iter_mut().zip(zp_correction.iter()) {
        *b -= corr;  // Subtract because: acc = Σ(a*w) - zp*Σ(w) = Σ(a*w) + bias_fused
    }
}
```

---

## 7. WASM Implementation

### 7.1 WASM SIMD128 Kernels

```rust
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WASM SIMD128 INT8 dot product
///
/// WASM SIMD128 doesn't have direct u8×i8→i16 like AVX2's maddubs.
/// Use i16x8 multiplication with widening.
#[cfg(target_arch = "wasm32")]
pub fn dot_product_int8_wasm(a: &[u8], b: &[i8]) -> i32 {
    let len = a.len();
    let chunks = len / 16;

    let mut acc = i32x4_splat(0);

    for i in 0..chunks {
        // Load 16 bytes each
        let va = v128_load(a.as_ptr().add(i * 16) as *const v128);
        let vb = v128_load(b.as_ptr().add(i * 16) as *const v128);

        // Widen to i16: process low and high halves
        // u8 -> i16 (zero extend)
        let a_lo = i16x8_extend_low_u8x16(va);
        let a_hi = i16x8_extend_high_u8x16(va);

        // i8 -> i16 (sign extend)
        let b_lo = i16x8_extend_low_i8x16(vb);
        let b_hi = i16x8_extend_high_i8x16(vb);

        // i16 × i16 -> i32
        let prod_lo = i32x4_extmul_low_i16x8(a_lo, b_lo);
        let prod_hi = i32x4_extmul_high_i16x8(a_lo, b_lo);
        let prod_lo2 = i32x4_extmul_low_i16x8(a_hi, b_hi);
        let prod_hi2 = i32x4_extmul_high_i16x8(a_hi, b_hi);

        // Accumulate
        acc = i32x4_add(acc, prod_lo);
        acc = i32x4_add(acc, prod_hi);
        acc = i32x4_add(acc, prod_lo2);
        acc = i32x4_add(acc, prod_hi2);
    }

    // Horizontal sum
    let sum = i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc);

    // Scalar remainder
    let mut result = sum;
    for i in (chunks * 16)..len {
        result += (a[i] as i32) * (b[i] as i32);
    }

    result
}
```

### 7.2 WASM Bindings

```rust
// In crates/ruvector-cnn-wasm/src/quantization.rs

#[wasm_bindgen]
pub struct QuantizedCnnWasm {
    #[wasm_bindgen(skip)]
    model: QuantizedMobileNetV3,
}

#[wasm_bindgen]
impl QuantizedCnnWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<QuantizedCnnWasm, JsValue> {
        let model = QuantizedMobileNetV3::from_bytes(weights)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { model })
    }

    /// Get embedding for an image (HWC u8 format)
    pub fn embed(&self, image: &[u8], width: usize, height: usize) -> Result<Vec<f32>, JsValue> {
        let input = preprocess_image(image, width, height)?;
        let embedding = self.model.forward(&input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(embedding)
    }

    /// Get model size info
    #[wasm_bindgen(getter, js_name = sizeBytes)]
    pub fn size_bytes(&self) -> usize {
        self.model.size_bytes()
    }

    /// Get embedding dimension
    #[wasm_bindgen(getter, js_name = embeddingDim)]
    pub fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }
}
```

---

## 8. Integration with Existing Crates

### 8.1 Integration Map

| Existing Component | File | Integration | Change Type |
|-------------------|------|-------------|-------------|
| Conv2d | `layers/conv.rs` | Add `to_quantized()` method | Extend |
| BatchNorm | `layers/batch_norm.rs` | Add `fuse_into(conv)` method | Extend |
| Activation | `layers/activation.rs` | Add INT8 variants | Extend |
| MobileNetV3 | `models/mobilenet.rs` | Add `QuantizedMobileNetV3` | Extend |
| SIMD dispatch | `simd/mod.rs` | Add INT8 kernel selection | Extend |
| AVX2 kernels | `simd/avx2.rs` | Add INT8 dot/conv kernels | Extend |
| NEON kernels | `simd/neon.rs` | Add INT8 kernels | Extend |
| Feature flags | `Cargo.toml` | Add `int8` feature | Extend |

### 8.2 Feature Gating

```toml
# In crates/ruvector-cnn/Cargo.toml

[features]
default = ["simd"]
simd = []
int8 = []                    # INT8 quantization support
int8-avx2 = ["int8"]        # AVX2 INT8 kernels
int8-neon = ["int8"]        # NEON INT8 kernels
int8-wasm = ["int8"]        # WASM SIMD INT8 kernels
calibration = ["int8"]      # Calibration infrastructure
```

---

## 9. Implementation Timeline

| Week | Phase | Deliverables | Key Files |
|------|-------|-------------|-----------|
| 1 | **Core Types** | QuantParams, QuantizedTensor, scale computation | `quantization/params.rs`, `tensor.rs`, `ops.rs` |
| 2 | **AVX2 Kernels** | INT8 dot product, 3×3 conv | `quantization/simd/avx2.rs` |
| 3 | **Quantized Layers** | QuantizedConv2d, fused BatchNorm | `quantization/layers/conv.rs`, `fused.rs` |
| 4 | **Activations** | INT8 ReLU, ReLU6, HardSwish | `quantization/layers/activation.rs` |
| 5 | **Calibration** | Stats collection, MinMax/Percentile | `quantization/calibration/` |
| 6 | **Model Wrapper** | QuantizedMobileNetV3, end-to-end | `quantization/model.rs` |
| 7 | **NEON/WASM** | Platform kernels, WASM bindings | `simd/neon.rs`, `simd/wasm.rs` |
| 8 | **Benchmarks & Validation** | Criterion suite, accuracy tests | `benches/int8_quant_bench.rs` |

---

## 10. Success Criteria

### 10.1 Quantitative Targets

| Metric | Target | Method |
|--------|--------|--------|
| MobileNetV3-Small speedup | ≥2.5x vs FP32 | Criterion benchmark |
| MobileNetV3-Small latency | <5 ms (M4) | Criterion benchmark |
| Top-1 accuracy drop | <1% | ImageNet validation subset |
| Embedding cosine similarity | >0.995 vs FP32 | Test suite |
| INT8 3×3 conv throughput | >20 GOPS | Criterion benchmark |
| Calibration time (100 images) | <30 s | Integration test |
| WASM binary size increase | <50 KB | Build measurement |
| Memory reduction | ≥3x vs FP32 | Model size comparison |

### 10.2 Qualitative Criteria

- All existing FP32 tests continue to pass
- INT8 model produces identical output structure as FP32
- Calibration workflow is documented with examples
- SIMD kernels have scalar fallbacks for all platforms
- No unsafe code without safety documentation

---

## 11. Consequences

### 11.1 Positive

- **2-4x inference speedup**: Significant performance improvement for real-time applications
- **4x memory reduction**: INT8 weights/activations are 1 byte vs 4 bytes
- **Better cache utilization**: Smaller data fits in L1/L2 cache
- **Edge deployment**: Enables deployment on memory-constrained devices
- **Standard approach**: INT8 PTQ is well-understood, low risk

### 11.2 Negative

- **Calibration requirement**: Need representative data for accurate quantization
- **Slight accuracy loss**: <1% but non-zero degradation
- **Maintenance burden**: Additional kernel implementations per platform
- **Testing complexity**: Need to validate both FP32 and INT8 paths

### 11.3 Mitigations

- **Calibration data**: Provide default calibration from ImageNet subset
- **Accuracy validation**: Automated accuracy regression tests in CI
- **Kernel sharing**: Share patterns between AVX2/NEON/WASM implementations
- **Feature gating**: INT8 is opt-in; FP32 remains default

---

## 12. Related Decisions

- **ADR-090**: Ultra-Low-Bit QAT & Pi-Quantization (LLM quantization patterns)
- **ADR-003**: SIMD Optimization Strategy (existing SIMD infrastructure)
- **ADR-005**: WASM Runtime Integration (WASM deployment patterns)
- **ADR-001**: RuVector Core Architecture (overall system design)

---

## 13. References

- `crates/ruvector-cnn/docs/INT8_QUANTIZATION_DESIGN.md` — Detailed implementation design
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Google, 2018)
- "A Survey of Quantization Methods for Efficient Neural Network Inference" (Wu et al., 2021)
- Intel Intrinsics Guide: AVX2 integer operations
- PyTorch Quantization documentation
- TensorRT INT8 calibration documentation
