// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* rewards [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& gamma [[buffer(3)]],
    constant float& lam [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  float next_value = 0, advantage = 0;
  for (int t = int(steps) - 1; t >= 0; --t) {
    uint index = i * steps + t;
    advantage = rewards[index] + gamma * next_value - values[index] + gamma * lam * advantage;
    out[index] = advantage;
    next_value = values[index];
  }
}
