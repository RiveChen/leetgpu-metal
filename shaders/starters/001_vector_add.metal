#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
  // TODO: Write C[i] = A[i] + B[i] for 0 <= i < N.
}
