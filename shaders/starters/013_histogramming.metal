#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* input [[buffer(0)]],
    device atomic_int* histogram [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& num_bins [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
  // TODO: Count occurrences among N inputs for each bin from 0 to num_bins - 1.
}
