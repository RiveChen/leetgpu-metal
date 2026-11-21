// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 004: Reduction (Sum)
//
// Task:
// Input array has N elements. Compute the scalar sum and store it in output[0].
//
// Optimization Tips:
// 1. Grid-stride loop: Each thread accumulates multiple elements into a local register.
// 2. Intra-warp reduction: Use `simd_sum(val)` to reduce across each 32-thread SIMD group.
// 3. Threadgroup reduction: Share partial sums from each SIMD group in `threadgroup` memory.
// 4. For very large N, a multi-block / two-pass reduction can fully saturate GPU bandwidth.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // Step 1: Thread-local accumulation with stride
  float sum = 0.0f;
  for (uint i = tid; i < N; i += tg_size) {
    sum += input[i];
  }

  // Step 2: SIMD-level reduction (Apple Silicon SIMD group size is 32)
  sum = simd_sum(sum);

  // Step 3: Threadgroup-level reduction across SIMD groups
  threadgroup float shared_sums[32]; // Max 32 SIMD groups (up to 1024 threads)
  if (simd_lane == 0) {
    shared_sums[simd_id] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 4: Final reduction by the first SIMD group
  if (simd_id == 0) {
    uint num_simd_groups = (tg_size + 31) / 32;
    float final_sum = (simd_lane < num_simd_groups) ? shared_sums[simd_lane] : 0.0f;
    final_sum = simd_sum(final_sum);

    if (simd_lane == 0) {
      output[0] = final_sum;
    }
  }
}
