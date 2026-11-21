#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* trig [[buffer(2)]],
    device const int* positions [[buffer(3)]],
    device const float* old_keys [[buffer(4)]],
    device const float* old_values [[buffer(5)]],
    device float* query [[buffer(6)]],
    device float* keys [[buffer(7)]],
    device float* values [[buffer(8)]],
    constant uint& batch [[buffer(9)]],
    constant uint& model [[buffer(10)]],
    constant uint& q_heads [[buffer(11)]],
    constant uint& kv_heads [[buffer(12)]],
    constant uint& width [[buffer(13)]],
    constant uint& capacity [[buffer(14)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Project Q/K/V, rotate Q and K, and return Q and both updated caches.
}
