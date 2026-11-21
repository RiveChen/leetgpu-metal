#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
  // TODO: Normalize each row over C features using eps, then apply per-feature weight and bias.
}
