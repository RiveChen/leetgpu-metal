#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* alpha [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply the per-channel decayed delta-memory recurrence and query it after each update.
}
