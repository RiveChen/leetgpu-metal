// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device const int* labels [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& samples [[buffer(3)]],
    constant uint& classes [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  float total = 0;
  for (uint row = 0; row < samples; ++row) {
    device const float* scores = logits + row * classes;
    float largest = scores[0];
    for (uint j = 1; j < classes; ++j)
      largest = max(largest, scores[j]);
    float sum = 0;
    for (uint j = 0; j < classes; ++j)
      sum += exp(scores[j] - largest);
    total += log(sum) + (largest - scores[labels[row]]);
  }
  loss[0] = total / float(samples);
}
