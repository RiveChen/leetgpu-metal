#include <metal_stdlib>
using namespace metal;

// Problem 042: 2D Max Pooling (2x2 kernel, stride 2)
//
// Starter Framework:
// Input: (H, W), Output: (H / 2, W / 2). Each thread samples a 2x2 patch.
//
// Optimization Tips:
// 1. Threadgroup Memory Caching: Cache adjacent 2D patches to eliminate redundant loads.
// 2. Vectorized 2D Reads: Load two contiguous elements simultaneously to minimize memory transactions.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& H [[buffer(2)]],
    constant uint& W [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
  uint c = gid.x;
  uint r = gid.y;
  uint out_H = H / 2;
  uint out_W = W / 2;

  if (r < out_H && c < out_W) {
    uint in_r = r * 2;
    uint in_c = c * 2;

    float v00 = input[in_r * W + in_c];
    float v01 = input[in_r * W + (in_c + 1)];
    float v10 = input[(in_r + 1) * W + in_c];
    float v11 = input[(in_r + 1) * W + (in_c + 1)];

    output[r * out_W + c] = max(max(v00, v01), max(v10, v11));
  }
}
