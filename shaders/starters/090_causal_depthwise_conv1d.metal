#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& taps [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply a causal filter per channel. Tap zero multiplies the current timestep.
}
