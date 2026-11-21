#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Return the first index attaining the maximum in each row.
}
