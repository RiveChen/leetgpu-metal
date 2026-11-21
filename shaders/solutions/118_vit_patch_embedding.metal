// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* images [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* cls [[buffer(3)]],
    device const float* pos [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& height [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant uint& patch [[buffer(10)]],
    constant uint& embed [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  uint across = width / patch, tokens = (height / patch) * across + 1;
  if (i >= batch * tokens * embed)
    return;
  uint b = i / (tokens * embed), t = (i / embed) % tokens, d = i % embed;
  float value = cls[d];
  if (t) {
    value = bias[d];
    uint row = (t - 1) / across * patch, col = (t - 1) % across * patch;
    for (uint c = 0; c < channels; ++c)
      for (uint y = 0; y < patch; ++y)
        for (uint x = 0; x < patch; ++x)
          value += images[((b * channels + c) * height + row + y) * width + col + x] *
                   weights[((d * channels + c) * patch + y) * patch + x];
  }
  output[i] = value + pos[t * embed + d];
}
