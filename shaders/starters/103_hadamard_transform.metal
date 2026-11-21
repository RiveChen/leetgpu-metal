#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply an orthonormal Walsh-Hadamard transform to each power-of-two row.
}
