#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const uchar* idx [[buffer(1)]],
    device const char* signs [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* rotation [[buffer(4)]],
    device const float* projection [[buffer(5)]],
    device const float* book [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    constant uint& length [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    constant uint& levels [[buffer(11)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
  // TODO: Compute query scores against codebook keys with inverse rotation and QJL residual correction.
}
