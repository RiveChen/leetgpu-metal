#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* h [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute h[t] = a[t] h[t-1] + x[t] independently for each sequence, initially h = 0.
}
