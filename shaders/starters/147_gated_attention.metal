#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* gate [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& model [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Gate the causal multi-head attention output elementwise using sigmoid(X @ Wg).
}
