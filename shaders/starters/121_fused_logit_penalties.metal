#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device const int* prompt [[buffer(1)]],
    device const int* generated [[buffer(2)]],
    device const float* presence [[buffer(3)]],
    device const float* frequency [[buffer(4)]],
    device const float* repetition [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& vocab [[buffer(8)]],
    constant uint& prompt_len [[buffer(9)]],
    constant uint& generated_len [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply repetition, frequency and presence penalties per token, ignoring -1 padding.
}
