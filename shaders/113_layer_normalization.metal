#include <metal_stdlib>
using namespace metal;

// Problem 113: Layer Normalization (LayerNorm)
//
// Starter Framework:
// Standard LayerNorm across feature dimension C for N rows:
// 1. mean = mean(x)
// 2. var = var(x)
// 3. y = weight * (x - mean) / sqrt(var + eps) + bias
//
// Optimization Tips:
// 1. Welford's Algorithm: Compute both mean and variance in a single pass to save memory traffic.
// 2. Intra-SIMD reductions: Use `simd_sum` to rapidly aggregate mean and variance within 32-thread warps.
// 3. Vectorized weight/bias access: Cache affine parameters in threadgroup memory or registers.

kernel void solve(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  uint row = tgid;
  if (row >= N)
    return;

  device const float* row_in = input + row * C;
  device float* row_out = output + row * C;

  threadgroup float shared_buf[32];
  threadgroup float shared_mean;
  threadgroup float shared_inv_std;

  // Pass 1: Compute row mean
  float local_sum = 0.0f;
  for (uint c = tid; c < C; c += tg_size) {
    local_sum += row_in[c];
  }
  local_sum = simd_sum(local_sum);

  if (simd_lane == 0) {
    shared_buf[simd_id] = local_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float m = (simd_lane < num_groups) ? shared_buf[simd_lane] : 0.0f;
    m = simd_sum(m);
    if (simd_lane == 0) {
      shared_mean = m / float(C);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = shared_mean;

  // Pass 2: Compute variance
  float local_sq = 0.0f;
  for (uint c = tid; c < C; c += tg_size) {
    float diff = row_in[c] - mean;
    local_sq = fma(diff, diff, local_sq);
  }
  local_sq = simd_sum(local_sq);

  if (simd_lane == 0) {
    shared_buf[simd_id] = local_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0) {
    uint num_groups = (tg_size + 31) / 32;
    float v = (simd_lane < num_groups) ? shared_buf[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0) {
      float var = v / float(C);
      shared_inv_std = rsqrt(var + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_std = shared_inv_std;

  // Pass 3: Normalize and affine scale/shift
  for (uint c = tid; c < C; c += tg_size) {
    float norm = (row_in[c] - mean) * inv_std;
    row_out[c] = fma(weight[c], norm, bias[c]);
  }
}
