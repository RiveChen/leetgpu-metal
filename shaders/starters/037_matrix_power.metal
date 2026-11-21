#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& power [[buffer(3)]],
    device float* workspace [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Raise the square matrix to a positive integer power. Workspace holds n*n floats.
}
