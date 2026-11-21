#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* flags [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute exclusive sums, resetting at each flagged segment start.
}
