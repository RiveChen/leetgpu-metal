#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const uchar* x [[buffer(0)]],
    device const float* xscale [[buffer(1)]],
    device const uchar* weights [[buffer(2)]],
    device const float* wscale [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Decode finite E4M3 values, apply each 128-element block scale and store the GEMM in FP16.
}
