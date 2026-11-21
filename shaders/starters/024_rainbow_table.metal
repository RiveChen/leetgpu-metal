#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device uint* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& rounds [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply R rounds of 32-bit FNV-1a to each word, least-significant byte first.
}
