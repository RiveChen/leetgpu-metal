#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* conditioning [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply the packed DiT block with conditioned layer norms, 8-head attention and GELU MLP.
}
