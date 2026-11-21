#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& H [[buffer(2)]],
    constant uint& W [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
  // TODO: Max-pool the H x W input using 2 x 2 windows and stride 2.
}
