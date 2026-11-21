#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* ids [[buffer(0)]],
    device const int* pos [[buffer(1)]],
    device const float* word [[buffer(2)]],
    device const float* position [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& vocab [[buffer(9)]],
    constant uint& positions [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Add token and position embeddings, then layer-normalize each token and apply gamma and beta.
}
