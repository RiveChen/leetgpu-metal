#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* input [[buffer(0)]],
    device const float* filter [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& W [[buffer(4)]],
    constant uint& kH [[buffer(5)]],
    constant uint& kW [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
  // TODO: Compute valid 2D cross-correlation. The output shape is (H - kH + 1, W - kW + 1).
}
