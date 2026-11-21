#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device float* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sort the n float32 values in ascending order in place.
}
