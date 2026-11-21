#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* params [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const float* m [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* params_out [[buffer(4)]],
    device float* m_out [[buffer(5)]],
    device float* v_out [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant float& lr [[buffer(8)]],
    constant float& beta1 [[buffer(9)]],
    constant float& beta2 [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant float& decay [[buffer(12)]],
    constant uint& step [[buffer(13)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute one AdamW step and return updated parameters and both moments.
}
