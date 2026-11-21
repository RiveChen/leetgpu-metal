#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* tokens [[buffer(0)]],
    device const float* draft [[buffer(1)]],
    device const float* target [[buffer(2)]],
    device const float* samples [[buffer(3)]],
    device int* out [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Verify draft tokens in order; draw a replacement at rejection or a bonus after full acceptance.
}
