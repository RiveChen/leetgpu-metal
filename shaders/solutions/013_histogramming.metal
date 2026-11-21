// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 013: Histogramming
//
// Task:
// Compute frequency of each bin (0 to num_bins - 1) from N integer values.
//
// Optimization Tips:
// 1. Privatization / Local Histogram: Allocate a `threadgroup atomic_int local_hist[256]`
//    in threadgroup memory. Accumulate locally to avoid heavy contention on device memory,
//    then flush the local histogram to global device memory with atomic_fetch_add.

kernel void solve(
    device const int* input [[buffer(0)]],
    device atomic_int* histogram [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& num_bins [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
  // Step 1: Initialize global histogram bins to 0
  for (uint b = tid; b < num_bins; b += tg_size) {
    atomic_store_explicit(&histogram[b], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Step 2: Accumulate counts into the global histogram
  for (uint i = tid; i < N; i += tg_size) {
    int val = input[i];
    if (val >= 0 && (uint)val < num_bins) {
      atomic_fetch_add_explicit(&histogram[val], 1, memory_order_relaxed);
    }
  }
}
