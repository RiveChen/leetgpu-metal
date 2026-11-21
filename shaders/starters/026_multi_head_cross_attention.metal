#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& head_count [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute scaled softmax attention from each query head over the n key/value rows in the same head.
}
