// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* ids [[buffer(0)]],
    device const int* pos [[buffer(1)]],
    device const float* word [[buffer(2)]],
    device const float* position [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& vocab [[buffer(9)]],
    constant uint& positions [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    uint i [[thread_position_in_grid]]
) {
  uint w = ids[i] * width, p = pos[i % length] * width;
  float mean = 0, var = 0;
  for (uint j = 0; j < width; ++j)
    mean += word[w + j] + position[p + j];
  mean /= float(width);
  for (uint j = 0; j < width; ++j) {
    float d = (word[w + j] + position[p + j]) - mean;
    var += d * d;
  }
  float inv = rsqrt(var / float(width) + eps);
  for (uint j = 0; j < width; ++j)
    out[i * width + j] = ((word[w + j] + position[p + j]) - mean) * inv * gamma[j] + beta[j];
}
