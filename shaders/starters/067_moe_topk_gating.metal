#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device float* weights [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: For each row, return descending top-k expert indices and their normalized softmax weights.
}
