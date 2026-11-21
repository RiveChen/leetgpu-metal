// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& tile [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows * cols)
    return;
  uint tile_row = (i / cols) / tile, tile_col = (i % cols) / tile;
  y[i] = x[i] * scale[tile_row * ((cols + tile - 1) / tile) + tile_col];
}
