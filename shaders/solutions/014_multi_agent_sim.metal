// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float4* agents [[buffer(0)]],
    device float4* next_agents [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n)
    return;
  float4 agent = agents[i];
  float2 sum = 0;
  uint count = 0;
  for (uint j = 0; j < n; ++j) {
    float2 delta = agent.xy - agents[j].xy;
    if (j != i && dot(delta, delta) < 25) {
      sum += agents[j].zw;
      ++count;
    }
  }
  float2 velocity = agent.zw;
  if (count)
    velocity += 0.05f * (sum / float(count) - velocity);
  next_agents[i] = float4(agent.xy + velocity, velocity);
}
