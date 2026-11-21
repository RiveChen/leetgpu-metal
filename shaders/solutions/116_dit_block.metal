// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

constant uint D = 512;
inline float linear(
    device const float* x,
    device const float* weights,
    device const float* bias,
    uint row,
    uint width
) {
  float sum = bias[row];
  for (uint j = 0; j < width; ++j)
    sum += x[j] * weights[row * width + j];
  return sum;
}

inline void modulated_norm(
    device const float* x, device float* out, device const float* mod, uint length, uint lane
) {
  for (uint t = lane; t < length; t += 64) {
    device const float* row = x + t * D;
    float mean = 0;
    for (uint j = 0; j < D; ++j)
      mean += row[j];
    mean /= D;
    float variance = 0;
    for (uint j = 0; j < D; ++j) {
      float z = row[j] - mean;
      variance += z * z;
    }
    float inv = rsqrt(variance / D + 1e-6f);
    for (uint j = 0; j < D; ++j)
      out[t * D + j] = (row[j] - mean) * inv * (1 + mod[D + j]) + mod[j];
  }
}

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* conditioning [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  uint b = i / 64, lane = i % 64;
  uint size = length * D;
  device float* mod = scratch + b * (6 * D + 10 * size);
  device float* qkv = mod + 6 * D;
  device float* normalized = qkv + 3 * size;
  device float* attention = normalized + size;
  device float* hidden = attention + size;
  device float* residual = hidden + 4 * size;
  device const float* sample = x + b * size;
  device const float* cond = conditioning + b * D;
  device const float* ada_w = weights;
  device const float* ada_b = ada_w + 6 * D * D;
  device const float* qkv_w = ada_b + 6 * D;
  device const float* qkv_b = qkv_w + 3 * D * D;
  device const float* out_w = qkv_b + 3 * D;
  device const float* out_b = out_w + D * D;
  device const float* mlp_w = out_b + D;
  device const float* mlp_b = mlp_w + 4 * D * D;
  device const float* down_w = mlp_b + 4 * D;
  device const float* down_b = down_w + 4 * D * D;
  // Reuse the normalized-token buffer to hold SiLU(c) during modulation.
  for (uint j = lane; j < D; j += 64)
    normalized[j] = cond[j] / (1 + exp(-cond[j]));
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < 6 * D; j += 64)
    mod[j] = linear(normalized, ada_w, ada_b, j, D);
  threadgroup_barrier(mem_flags::mem_device);
  modulated_norm(sample, normalized, mod, length, lane);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < 3 * size; j += 64)
    qkv[j] = linear(normalized + (j / (3 * D)) * D, qkv_w, qkv_b, j % (3 * D), D);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < size; j += 64) {
    uint t = j / D, d = j % D, head = d / 64;
    device const float* q = qkv + t * 3 * D + head * 64;
    float peak = -INFINITY, den = 0, total = 0;
    for (uint s = 0; s < length; ++s) {
      device const float* k = qkv + s * 3 * D + D + head * 64;
      float score = 0;
      for (uint kdim = 0; kdim < 64; ++kdim)
        score += q[kdim] * k[kdim];
      score *= 0.125f;
      float next = max(score, peak), old = exp(peak - next), w = exp(score - next);
      total = total * old + w * qkv[s * 3 * D + 2 * D + d];
      den = den * old + w;
      peak = next;
    }
    attention[j] = total / den;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < size; j += 64) {
    uint d = j % D;
    residual[j] = sample[j] + mod[2 * D + d] * linear(attention + (j - d), out_w, out_b, d, D);
  }
  threadgroup_barrier(mem_flags::mem_device);
  modulated_norm(residual, normalized, mod + 3 * D, length, lane);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < 4 * size; j += 64) {
    float z = linear(normalized + j / (4 * D) * D, mlp_w, mlp_b, j % (4 * D), D);
    hidden[j] = 0.5f * z * (1 + tanh(0.79788456f * (z + 0.044715f * z * z * z)));
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint j = lane; j < size; j += 64) {
    uint d = j % D;
    output[b * size + j] =
        residual[j] + mod[5 * D + d] * linear(hidden + (j - d) * 4, down_w, down_b, d, 4 * D);
  }
}
