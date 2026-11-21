#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& nnz [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply the sparse matrix stored in row-major dense layout by x.
}
