// AI-generated reference solution.
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
  if (i >= batch * vocab)
    return;
  uint b = i / vocab;
  int token = int(i % vocab);
  uint count = 0;
  for (uint j = 0; j < generated_len; ++j)
    count += uint(generated[b * generated_len + j] == token);
  bool seen = count > 0;
  for (uint j = 0; j < prompt_len && !seen; ++j)
    seen = prompt[b * prompt_len + j] == token;
  float x = logits[i];
  if (seen)
    x = x > 0 ? x / repetition[b] : x * repetition[b];
  output[i] = x - float(count) * frequency[b] - (count ? presence[b] : 0.0f);
}
