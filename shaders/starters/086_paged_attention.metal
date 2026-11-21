#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* lengths [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& block [[buffer(9)]],
    constant uint& pages [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Gather valid KV entries through the block table and evaluate scaled dot-product attention.
}
