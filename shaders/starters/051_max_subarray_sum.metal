#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& window [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Return the largest sum of any contiguous window of exactly the specified length.
}
