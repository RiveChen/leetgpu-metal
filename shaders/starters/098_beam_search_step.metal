#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* previous [[buffer(0)]],
    device const float* logp [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device int* parents [[buffer(3)]],
    device int* tokens [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& beams [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Select the best K distinct-scoring beam/token expansions per batch in descending order.
}
