// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* rstd [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dx [[buffer(5)]],
    device float* dgamma [[buffer(6)]],
    device float* dbeta [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& channels [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= channels)
    return;
  float dg = 0, db = 0;
  for (uint row = 0; row < rows; ++row) {
    uint base = row * channels;
    float sum = 0, dot = 0;
    for (uint c = 0; c < channels; ++c) {
      float g = dy[base + c] * gamma[c];
      sum += g;
      dot += g * (x[base + c] - mean[row]) * rstd[row];
    }
    float h = (x[base + i] - mean[row]) * rstd[row], d = dy[base + i];
    dx[base + i] = rstd[row] * (d * gamma[i] - (sum + h * dot) / float(channels));
    dg += d * h;
    db += d;
  }
  dgamma[i] = dg;
  dbeta[i] = db;
}
