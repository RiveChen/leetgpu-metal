// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    device float* dweight [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= channels)
    return;
  float dw = 0;
  for (uint row = 0; row < rows; ++row) {
    uint base = row * channels;
    float square_sum = 0, dot = 0;
    for (uint c = 0; c < channels; ++c) {
      float v = x[base + c];
      square_sum += v * v;
      dot += dy[base + c] * weight[c] * v;
    }
    float inv = rsqrt(square_sum / float(channels) + eps);
    float norm = x[base + i] * inv;
    dx[base + i] = inv * (dy[base + i] * weight[i] - norm * dot * inv / float(channels));
    dw += dy[base + i] * norm;
  }
  dweight[i] = dw;
}
