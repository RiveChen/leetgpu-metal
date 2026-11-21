#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* ids [[buffer(1)]],
    device const float* mix [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* up [[buffer(4)]],
    device const float* down [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    constant uint& hidden [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    constant uint& experts [[buffer(10)]],
    constant uint& top [[buffer(11)]],
    device float* scratch [[buffer(12)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Route each token to its selected experts, apply SwiGLU, and blend their projections.
}
