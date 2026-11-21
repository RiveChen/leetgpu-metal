// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& window [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  int sum = 0;
  for (uint j = 0; j < window; ++j)
    sum += x[j];
  int best = sum;
  for (uint j = window; j < n; ++j) {
    sum += x[j] - x[j - window];
    best = max(sum, best);
  }
  output[0] = best;
}
