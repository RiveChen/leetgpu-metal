#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const char* weights [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: RMS-normalize, quantize each row to symmetric int8, and project with per-output weight scales.
}
