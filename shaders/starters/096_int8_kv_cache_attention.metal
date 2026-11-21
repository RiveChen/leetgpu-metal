#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const char* keys [[buffer(1)]],
    device const char* values [[buffer(2)]],
    device const float* key_scale [[buffer(3)]],
    device const float* value_scale [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Dequantize the int8 key/value cache with per-token scales and compute attention per head.
}
