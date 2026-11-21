#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply full scaled attention; consecutive query heads share one KV head.
}
