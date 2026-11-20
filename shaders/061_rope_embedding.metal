#include <metal_stdlib>
using namespace metal;

// Problem 061: Rotary Position Embedding (RoPE)
//
// Starter Framework:
// Used in LLaMA, Mistral, Qwen, and DeepSeek architectures:
// x1 = Q[..., : D / 2], x2 = Q[..., D / 2 :]
// rotated_Q = [-x2, x1]
// output = Q * cos + rotated_Q * sin
//
// Optimization Tips:
// 1. Threadgroup Tiling / Register Caching: Pair indices (col, col + half) together to reuse loads.
// 2. FMA Operations: Combine multiplications and additions using `fma`.

kernel void solve(
    device const float* Q [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
  uint col = gid.x;
  uint row = gid.y;

  if (row < M && col < D) {
    uint d_half = D / 2;
    uint idx = row * D + col;

    float q_val = Q[idx];
    float cos_val = cos[idx];
    float sin_val = sin[idx];

    // Determine element from the complementary half
    float rot_val;
    if (col < d_half) {
      rot_val = -Q[row * D + (col + d_half)];
    } else {
      rot_val = Q[row * D + (col - d_half)];
    }

    output[idx] = fma(rot_val, sin_val, q_val * cos_val);
  }
}
