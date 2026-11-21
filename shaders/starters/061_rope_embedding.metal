#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* Q [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
  // TODO: Apply split-half rotary embeddings to each of the M rows using cos and sin.
}
