#include <metal_stdlib>
using namespace metal;

// Problem 017: Dot Product
//
// Starter Framework:
// Compute scalar output[0] = sum_{i=0}^{N-1} A[i] * B[i].
//
// Optimization Tips:
// 1. FMA (Fused Multiply-Add): Utilize `fma(a, b, c)` to execute multiply and accumulate in single instruction.
// 2. Vectorized Loads: Read `float4` vectors to saturate memory bandwidth.
// 3. Multi-level Reduction: Reduce within SIMD group via `simd_sum`, then across threadgroups.

kernel void solve(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // Step 1: Thread-local FMA accumulation with stride
  float local_dot = 0.0f;
  for (uint i = tid; i < N; i += tg_size) {
    local_dot = fma(A[i], B[i], local_dot);
  }

  // Step 2: Intra-SIMD reduction
  local_dot = simd_sum(local_dot);

  // Step 3: Threadgroup reduction across SIMD groups
  threadgroup float shared_sums[32];
  if (simd_lane == 0) {
    shared_sums[simd_id] = local_dot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float final_sum = (simd_lane < num_groups) ? shared_sums[simd_lane] : 0.0f;
    final_sum = simd_sum(final_sum);

    if (simd_lane == 0) {
      output[0] = final_sum;
    }
  }
}
