// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* tokens [[buffer(0)]],
    device const float* draft [[buffer(1)]],
    device const float* target [[buffer(2)]],
    device const float* samples [[buffer(3)]],
    device int* out [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch)
    return;
  device int* result = out + i * (steps + 1);
  device const float* draws = samples + i * (steps + 1);
  for (uint j = 0; j <= steps; ++j)
    result[j] = 0;
  uint position = 0;
  for (; position < steps; ++position) {
    int token = tokens[i * steps + position];
    uint pos = (i * steps + position) * vocab + token;
    if (draws[position] >= min(1.0f, target[pos] / draft[pos]))
      break;
    result[position] = token;
  }
  bool bonus = position == steps;
  uint offset = (i * steps + min(position, steps - 1)) * vocab;
  float total = 0;
  for (uint j = 0; j < vocab; ++j)
    total += bonus ? target[offset + j] : max(0.0f, target[offset + j] - draft[offset + j]);
  float mass = 0, threshold = draws[steps] * (total > 0 ? total : float(vocab));
  for (uint j = 0; j < vocab; ++j) {
    mass += total == 0
                ? 1.0f
                : (bonus ? target[offset + j] : max(0.0f, target[offset + j] - draft[offset + j]));
    if (mass >= threshold || j + 1 == vocab) {
      result[position] = int(j);
      break;
    }
  }
}
