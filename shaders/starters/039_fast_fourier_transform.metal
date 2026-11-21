#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float2* signal [[buffer(0)]],
    device float2* spectrum [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute the unnormalized forward transform of the complex signal, stored as (real, imaginary) pairs. One threadgroup covers the entire signal.
}
