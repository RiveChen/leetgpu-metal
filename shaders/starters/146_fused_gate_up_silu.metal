#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Project through packed gate and up weights, then multiply SiLU(gate) by up.
}
