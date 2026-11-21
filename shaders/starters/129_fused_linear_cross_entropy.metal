#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* targets [[buffer(2)]],
    device float* loss [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& vocab [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute mean cross entropy from the linear projection without storing the logits.
}
