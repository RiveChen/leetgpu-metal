#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* values [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Write the k largest values in descending order, including repeated values.
}
