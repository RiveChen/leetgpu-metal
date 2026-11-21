#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& nnz [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply a sparse matrix in row-major dense storage by a dense matrix.
}
