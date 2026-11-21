#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const uchar* x [[buffer(0)]],
    device const uchar* xs [[buffer(1)]],
    device const uchar* w [[buffer(2)]],
    device const uchar* ws [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply packed high-nibble-first E2M1 operands with per-block E4M3 scales, then cast to half.
}
