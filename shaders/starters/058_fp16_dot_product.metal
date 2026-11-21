#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Store the FP16 dot product of the two vectors, accumulating in FP32.
}
