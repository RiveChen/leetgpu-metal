// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& depth [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& target [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  int partial = 0;
  for (uint j = i; j < depth * rows * cols; j += 64)
    partial += int(x[j] == target);
  threadgroup int sums[64];
  sums[i] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 32; step > 0; step /= 2) {
    if (i < step)
      sums[i] += sums[i + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i == 0)
    output[0] = sums[0];
}
