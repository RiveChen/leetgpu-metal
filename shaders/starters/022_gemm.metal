#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device half* c [[buffer(0)]],
    device const half* a [[buffer(1)]],
    device const half* b [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Update C in place with alpha * A @ B + beta * C, accumulating in FP32.
}
