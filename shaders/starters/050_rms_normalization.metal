#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& gamma [[buffer(2)]],
    constant float& beta [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // TODO: Apply RMS normalization across N elements with epsilon eps, scale gamma and bias beta.
}
