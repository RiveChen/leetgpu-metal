#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* mse [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // TODO: Write the mean squared difference of the N prediction-target pairs to mse[0].
}
