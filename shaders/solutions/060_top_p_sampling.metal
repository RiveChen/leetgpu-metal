// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline uint pcg(uint x) {
  uint state = x * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
  return word ^ (word >> 22);
}

kernel void solve(
    device const float* logits [[buffer(0)]],
    device const float* p [[buffer(1)]],
    device const int* seed [[buffer(2)]],
    device int* token [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  float maximum = logits[0];
  for (uint j = 1; j < n; ++j)
    maximum = max(maximum, logits[j]);
  float total = 0;
  for (uint j = 0; j < n; ++j)
    total += precise::exp(logits[j] - maximum);
  float cutoff = p[0] * total, best = -1;
  uint choice = 0, key = pcg(uint(seed[0]));
  for (uint index = 0; index < n; ++index) {
    float logit = logits[index];
    if (p[0] < 1) {
      // Count all tokens preceding this one in a stable descending sort.
      float preceding = 0;
      for (uint j = 0; j < n; ++j)
        if (logits[j] > logit || (logits[j] == logit && j < index))
          preceding += precise::exp(logits[j] - maximum);
      if (preceding >= cutoff)
        continue;
    }
    float weight = precise::exp(logit - maximum);
    uint bits = pcg(key + index);
    // Using the distance from one preserves precision for the largest hashes.
    float distance = (float(~bits) + 0.5f) * 0x1p-32f;
    float noise = bits < 0x80000000u ? -precise::log((float(bits) + 0.5f) * 0x1p-32f)
                                     : (distance < 1e-4f ? distance * (1 + 0.5f * distance)
                                                         : -precise::log(1 - distance));
    float score = weight / noise;
    if (score > best) {
      best = score;
      choice = index;
    }
  }
  token[0] = int(choice);
}
