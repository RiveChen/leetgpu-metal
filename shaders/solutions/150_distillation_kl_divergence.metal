// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* teacher [[buffer(0)]],
    device const float* student [[buffer(1)]],
    device float* kl [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows)
    return;
  uint base = i * vocab;
  float ma = teacher[base], mb = student[base];
  for (uint j = 1; j < vocab; ++j) {
    ma = max(ma, teacher[base + j]);
    mb = max(mb, student[base + j]);
  }
  float a = 0, b = 0, difference = 0;
  for (uint j = 0; j < vocab; ++j) {
    float u = (teacher[base + j] - ma) / temperature, v = (student[base + j] - mb) / temperature;
    float p = exp(u);
    a += p;
    b += exp(v);
    difference += p * (u - v);
  }
  kl[i] = difference / a + log(b) - log(a);
}
