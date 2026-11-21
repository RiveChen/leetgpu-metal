#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device const int* labels [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& samples [[buffer(3)]],
    constant uint& classes [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Write the mean categorical cross entropy from logits and zero-based class labels.
}
