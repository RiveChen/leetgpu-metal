// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float4* boxes [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device int* keep [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  threadgroup uint candidates[64];
  for (uint j = i; j < count; j += 64)
    keep[j] = -1;
  threadgroup_barrier(mem_flags::mem_device);
  while (true) {
    uint best = count;
    for (uint j = i; j < count; j += 64) {
      if (keep[j] < 0 && (best == count || scores[j] > scores[best]))
        best = j;
    }
    candidates[i] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 32; step; step /= 2) {
      if (i < step) {
        uint a = candidates[i], b = candidates[i + step];
        if (b < count && (a == count || scores[b] > scores[a] || (scores[b] == scores[a] && b < a)))
          candidates[i] = b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint winner = candidates[0];
    if (winner == count)
      break;
    float4 a = boxes[winner];
    float area = (a.z - a.x) * (a.w - a.y);
    for (uint j = i; j < count; j += 64) {
      if (j == winner) {
        keep[j] = 1;
        continue;
      }
      if (keep[j] >= 0)
        continue;
      float4 b = boxes[j];
      float2 size = max(min(a.zw, b.zw) - max(a.xy, b.xy), float2(0));
      float intersection = size.x * size.y;
      if (intersection / (area + (b.z - b.x) * (b.w - b.y) - intersection) > threshold)
        keep[j] = 0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}
