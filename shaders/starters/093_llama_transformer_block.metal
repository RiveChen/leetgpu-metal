#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* cosine [[buffer(2)]],
    device const float* sine [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  // TODO: Implement the packed-weight Llama block with RMSNorm, RoPE, causal GQA and SwiGLU.
}
