// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float2 embedding(int digit, device const float* w) {
  return float2(w[0] - w[1] * float(digit * digit), -float(digit));
}
inline float2 unit_rms(float2 v) {
  return v * rsqrt(.5f * dot(v, v) + 1e-6f);
}
inline float silu(float x) {
  return x / (1 + exp(-x));
}

kernel void solve(
    device const int* prompts [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch)
    return;
  int tokens[42];
  for (uint j = 0; j < 31; ++j)
    tokens[j] = prompts[i * 31 + j];
  const float omega = 2 * M_PI_F / 19;
  const float scale = log(10.0f) / (2 * (cos(.3f * omega) - cos(.7f * omega)));
  for (uint step = 0; step < 11; ++step) {
    uint last = 30 + step;
    float2 h = embedding(tokens[last], w);
    float2 q = unit_rms(unit_rms(h).x * float2(w[2], w[3]));
    float maximum = -INFINITY, numerator = 0, denominator = 0;
    for (uint j = 0; j <= last; ++j) {
      float2 normalized = unit_rms(embedding(tokens[j], w));
      float key = unit_rms(float2(normalized.x, 0)).x;
      float cosine;
      float sine = sincos(float(int(j) - int(last)) * omega, cosine);
      float score = scale * key * (q.x * cosine + q.y * sine);
      float next_max = max(maximum, score);
      float rescale = exp(maximum - next_max), weight = exp(score - next_max);
      denominator = denominator * rescale + weight;
      numerator = numerator * rescale + weight * (normalized.y * w[4]);
      maximum = next_max;
    }
    h.y += numerator / denominator;
    float2 normalized = unit_rms(h);
    float gate = dot(normalized, float2(w[5], w[6]));
    float other = gate - normalized.x * w[6] / 1000;
    h.y += w[7] * normalized.x * (silu(other) - silu(gate));
    float2 decoded = unit_rms(h) * float2(w[8], w[9]);
    float best = -INFINITY;
    uint next_token = 0;
    for (uint digit = 0; digit < 10; ++digit) {
      float score = dot(decoded, embedding(digit, w));
      logits[(i * 11 + step) * 10 + digit] = score;
      // Remove the class-independent offset before comparing close logits.
      float relative = -decoded.x * w[1] * float(digit * digit) - decoded.y * float(digit);
      if (relative > best) {
        best = relative;
        next_token = digit;
      }
    }
    tokens[last + 1] = next_token;
  }
}
