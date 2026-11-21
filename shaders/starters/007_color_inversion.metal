#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device uchar4* image [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Invert the RGB channels of each RGBA pixel in place and preserve alpha.
}
