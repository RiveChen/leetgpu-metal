#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float4* boxes [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device int* keep [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Write the greedy NMS keep mask; visit equal scores in index order.
}
