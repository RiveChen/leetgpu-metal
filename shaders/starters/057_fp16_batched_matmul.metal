#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute independent FP16 matrix products, accumulating in FP32.
}
