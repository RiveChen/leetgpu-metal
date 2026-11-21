#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& queries [[buffer(4)]],
    constant uint& keys [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Add alpha*(query_index-key_index) to each scaled dot product before row-wise softmax.
}
