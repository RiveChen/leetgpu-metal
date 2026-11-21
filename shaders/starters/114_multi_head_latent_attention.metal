#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* cache [[buffer(1)]],
    device const float* uk [[buffer(2)]],
    device const float* uv [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& rope [[buffer(9)]],
    device float* scratch [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
  // TODO: Absorb the key projection into the query, attend in latent space, then project values.
}
