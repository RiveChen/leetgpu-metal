// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 027: Mean Squared Error (MSE Loss)
//
// Task:
// Compute MSE = (1 / N) * sum_{i=0}^{N-1} (predictions[i] - targets[i])^2.
//
// Optimization Tips:
// 1. Threadgroup & SIMD reduction: Accumulate differences in registers before collective reduction.
// 2. Division hoisting: Divide by N only once at the final scalar write.

kernel void solve(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* mse [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  float sum_sq = 0.0f;
  for (uint i = tid; i < N; i += tg_size) {
    float diff = predictions[i] - targets[i];
    sum_sq = fma(diff, diff, sum_sq);
  }

  sum_sq = simd_sum(sum_sq);

  threadgroup float shared_sums[32];
  if (simd_lane == 0) {
    shared_sums[simd_id] = sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float final_sum = (simd_lane < num_groups) ? shared_sums[simd_lane] : 0.0f;
    final_sum = simd_sum(final_sum);

    if (simd_lane == 0) {
      mse[0] = final_sum / float(N);
    }
  }
}
