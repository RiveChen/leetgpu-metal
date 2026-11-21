// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 010: 2D Convolution (Cross-correlation with Valid Padding)
//
// Task:
// Input: (H, W), Kernel: (kH, kW), Output: (H - kH + 1, W - kW + 1)
//
// Optimization Tips:
// 1. Threadgroup Tiling / Shared Memory: Load an input tile (TILE + kH - 1, TILE + kW - 1)
//    into threadgroup memory to reuse overlapping window elements across adjacent threads.
// 2. Constant memory: Place small filter weights into constant memory address space (`constant float*`).

kernel void solve(
    device const float* input [[buffer(0)]],
    device const float* filter [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& W [[buffer(4)]],
    constant uint& kH [[buffer(5)]],
    constant uint& kW [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
  uint c = gid.x;
  uint r = gid.y;

  uint out_H = H - kH + 1;
  uint out_W = W - kW + 1;

  if (r < out_H && c < out_W) {
    float acc = 0.0f;
    for (uint kr = 0; kr < kH; ++kr) {
      for (uint kc = 0; kc < kW; ++kc) {
        acc += input[(r + kr) * W + (c + kc)] * filter[kr * kW + kc];
      }
    }
    output[r * out_W + c] = acc;
  }
}
