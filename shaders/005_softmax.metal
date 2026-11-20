#include <metal_stdlib>
using namespace metal;

// Problem 005: Softmax
//
// Starter Framework:
// Compute numerically stable softmax on 1D vector of length N:
// 1. max_val = max(x_i)
// 2. sum_exp = sum(exp(x_i - max_val))
// 3. y_i = exp(x_i - max_val) / sum_exp
//
// Optimization Tips:
// 1. Online Softmax: fuse the max reduction and sum-exp calculation into a single pass.
// 2. Intra-SIMD reductions: use `simd_max` and `simd_sum` to reduce communication latency.
// 3. Batched execution: for 2D inputs [Batch, Hidden], assign 1 threadgroup per row.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float shared_buf[32];
  threadgroup float shared_max;
  threadgroup float shared_sum;

  // Pass 1: Find global maximum (for numerical stability)
  float local_max = -INFINITY;
  for (uint i = tid; i < N; i += tg_size) {
    local_max = max(local_max, input[i]);
  }
  local_max = simd_max(local_max);

  if (simd_lane == 0) {
    shared_buf[simd_id] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float m = (simd_lane < num_groups) ? shared_buf[simd_lane] : -INFINITY;
    m = simd_max(m);
    if (simd_lane == 0) {
      shared_max = m;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float max_val = shared_max;

  // Pass 2: Compute sum of exponents
  float local_sum = 0.0f;
  for (uint i = tid; i < N; i += tg_size) {
    local_sum += exp(input[i] - max_val);
  }
  local_sum = simd_sum(local_sum);

  if (simd_lane == 0) {
    shared_buf[simd_id] = local_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float s = (simd_lane < num_groups) ? shared_buf[simd_lane] : 0.0f;
    s = simd_sum(s);
    if (simd_lane == 0) {
      shared_sum = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float sum_exp = shared_sum;

  // Pass 3: Normalize and store
  for (uint i = tid; i < N; i += tg_size) {
    output[i] = exp(input[i] - max_val) / sum_exp;
  }
}
