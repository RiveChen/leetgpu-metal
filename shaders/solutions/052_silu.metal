// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 052: SiLU (Sigmoid Linear Unit / Swish)
//
// Task:
// Compute y_i = x_i * sigmoid(x_i) = x_i / (1.0 + exp(-x_i)).
// Widely used in modern LLMs (LLaMA, Mistral, Qwen) and Diffusion Models.
//
// Optimization Tips:
// 1. Vectorized Memory Access: Process 4 elements per thread with `float4` and native Metal `precise::exp`.
// 2. Fast Math Mode: Enable fast-math to leverage Apple GPU native hardware reciprocal and transcendentals.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid < N) {
    float x = input[gid];
    float sig = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sig;
  }
}
