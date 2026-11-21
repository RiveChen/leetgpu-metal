// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& first_i [[buffer(3)]],
    constant uint& last_i [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  uint size_i = last_i - first_i + 1;
  int partial = 0;
  for (uint j = i; j < size_i; j += 64)
    partial += x[first_i + j % size_i];
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
