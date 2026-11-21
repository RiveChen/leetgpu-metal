#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* teacher [[buffer(0)]],
    device const float* student [[buffer(1)]],
    device float* kl [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute row-wise KL(teacher || student) between temperature-scaled logits.
}
