#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // TODO: Write the dot product of the two N-element vectors to output[0].
}
