#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* rewards [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& gamma [[buffer(3)]],
    constant float& lam [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Scan generalized advantages backward, with terminal value and terminal advantage equal to zero.
}
