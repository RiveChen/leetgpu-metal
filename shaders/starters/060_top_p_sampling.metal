#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device const float* p [[buffer(1)]],
    device const int* seed [[buffer(2)]],
    device int* token [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sample from the stable descending top-p nucleus using the per-token PCG exponential race.
}
