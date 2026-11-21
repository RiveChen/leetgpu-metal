// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& depth [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& first_z [[buffer(5)]],
    constant uint& last_z [[buffer(6)]],
    constant uint& first_row [[buffer(7)]],
    constant uint& last_row [[buffer(8)]],
    constant uint& first_col [[buffer(9)]],
    constant uint& last_col [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  uint size_z = last_z - first_z + 1;
  uint size_row = last_row - first_row + 1;
  uint size_col = last_col - first_col + 1;
  int partial = 0;
  for (uint j = i; j < size_z * size_row * size_col; j += 64)
    partial +=
        x[((first_z + (j / (size_row * size_col)) % size_z) * rows +
           (first_row + (j / (size_col)) % size_row)) *
              cols +
          (first_col + j % size_col)];
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
