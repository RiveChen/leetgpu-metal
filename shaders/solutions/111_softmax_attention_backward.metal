// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float inner(device const float* left, device const float* right, uint width) {
  float result = 0;
  for (uint d = 0; d < width; ++d)
    result += left[d] * right[d];
  return result;
}

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* grad [[buffer(3)]],
    device float* dq [[buffer(4)]],
    device float* dk [[buffer(5)]],
    device float* dv [[buffer(6)]],
    constant uint& queries [[buffer(7)]],
    constant uint& keys [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  for (uint j = 0; j < keys * width; ++j) {
    dk[j] = 0;
    dv[j] = 0;
  }
  float scale = rsqrt(float(width));
  for (uint row = 0; row < queries; ++row) {
    device const float* query = q + row * width;
    device const float* upstream = grad + row * width;
    float maximum = -INFINITY, sum = 0, weighted_dp = 0;
    for (uint j = 0; j < keys; ++j) {
      float score = inner(query, k + j * width, width) * scale, m = max(maximum, score);
      float rescale = exp(maximum - m), w = exp(score - m);
      sum = sum * rescale + w;
      weighted_dp = weighted_dp * rescale + w * inner(upstream, v + j * width, width);
      maximum = m;
    }
    for (uint d = 0; d < width; ++d)
      dq[row * width + d] = 0;
    for (uint j = 0; j < keys; ++j) {
      float p = exp(inner(query, k + j * width, width) * scale - maximum) / sum;
      float ds = p * (inner(upstream, v + j * width, width) - weighted_dp / sum) * scale;
      for (uint d = 0; d < width; ++d) {
        uint index = j * width + d;
        dq[row * width + d] += ds * k[index];
        dk[index] += ds * query[d];
        dv[index] += p * upstream[d];
      }
    }
  }
}
