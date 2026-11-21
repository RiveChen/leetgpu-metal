#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const uint* keys [[buffer(0)]],
    device uint* sorted [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    device uint* scratch [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Radix-sort the n unsigned 32-bit keys into ascending order. scratch holds n intermediate keys.
}
