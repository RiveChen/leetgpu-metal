// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  float partial = 0;
  for (uint j = i; j < n; j += 64)
    partial += float(a[j]) * float(b[j]);
  threadgroup float sums[64];
  sums[i] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 32; step > 0; step /= 2) {
    if (i < step)
      sums[i] += sums[i + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i == 0)
    output[0] = half(sums[0]);
}
