// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 050: RMS Normalization (Root Mean Square Normalization)
//
// Task:
// Used extensively in modern LLM architectures (e.g. LLaMA, Mistral, Qwen):
// 1. rms = sqrt(mean(x_i^2) + eps)
// 2. output_i = gamma * (x_i / rms) + beta
//
// Optimization Tips:
// 1. Fast reciprocal square root: use `rsqrt(mean_sq + eps)` to replace `1.0 / sqrt(...)`.
// 2. Intra-SIMD reductions: use `simd_sum` to compute partial sum of squares rapidly.
// 3. Batched execution: for [Batch, Hidden], assign 1 threadgroup per token vector.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& gamma [[buffer(2)]],
    constant float& beta [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // Step 1: Thread-local accumulation of sum of squares
  float sum_sq = 0.0f;
  for (uint i = tid; i < N; i += tg_size) {
    float x = input[i];
    sum_sq = fma(x, x, sum_sq);
  }

  // Step 2: SIMD-level reduction
  sum_sq = simd_sum(sum_sq);

  // Step 3: Threadgroup-level reduction
  threadgroup float shared_buf[32];
  threadgroup float shared_inv_rms;

  if (simd_lane == 0) {
    shared_buf[simd_id] = sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float total_sq = (simd_lane < num_groups) ? shared_buf[simd_lane] : 0.0f;
    total_sq = simd_sum(total_sq);

    if (simd_lane == 0) {
      float mean_sq = total_sq / float(N);
      shared_inv_rms = rsqrt(mean_sq + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_rms = shared_inv_rms;

  // Step 4: Scale, normalize, and write output
  for (uint i = tid; i < N; i += tg_size) {
    output[i] = fma(gamma, input[i] * inv_rms, beta);
  }
}
