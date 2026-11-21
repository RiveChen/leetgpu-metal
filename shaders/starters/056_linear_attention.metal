#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
  // TODO: Compute normalized linear attention with the ELU + 1 feature map.
}
