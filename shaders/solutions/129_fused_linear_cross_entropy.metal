// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* targets [[buffer(2)]],
    device float* loss [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& vocab [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  threadgroup float totals[64];
  float subtotal = 0;
  for (uint r = i; r < rows; r += 64) {
    float peak = -INFINITY, den = 0, target = 0;
    for (uint j = 0; j < vocab; ++j) {
      float logit = 0;
      for (uint k = 0; k < width; ++k)
        logit += x[r * width + k] * weights[j * width + k];
      if (j == uint(targets[r]))
        target = logit;
      float next = max(logit, peak);
      den = den * exp(peak - next) + exp(logit - next);
      peak = next;
    }
    subtotal += (peak - target) + log(den);
  }
  totals[i] = subtotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride; stride /= 2) {
    if (i < stride)
      totals[i] += totals[i + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i == 0)
    loss[0] = totals[0] / rows;
}
