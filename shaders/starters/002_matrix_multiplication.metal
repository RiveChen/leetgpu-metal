#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
  // TODO: Multiply row-major A (M x K) and B (K x N) into C (M x N).
}
