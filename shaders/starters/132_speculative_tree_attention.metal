#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* cache_k [[buffer(1)]],
    device const float* cache_v [[buffer(2)]],
    device const float* tree_k [[buffer(3)]],
    device const float* tree_v [[buffer(4)]],
    device const int* parents [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& prefix [[buffer(7)]],
    constant uint& nodes [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Attend to the shared prefix and the current tree node’s ancestor chain.
}
