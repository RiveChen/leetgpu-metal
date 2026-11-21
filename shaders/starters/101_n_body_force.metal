#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* positions [[buffer(0)]],
    device const float* masses [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& particles [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sum mass-weighted inverse-square interactions in 3D using softening epsilon 0.001.
}
