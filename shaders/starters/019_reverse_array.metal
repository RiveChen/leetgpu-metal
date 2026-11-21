#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Reverse the N elements in place.
}
