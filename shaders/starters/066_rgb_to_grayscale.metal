#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* rgb [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Convert packed RGB pixels to grayscale with weights 0.299, 0.587 and 0.114.
}
