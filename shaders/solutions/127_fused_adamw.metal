// AI-generated reference solution.
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
  if (i >= n)
    return;
  float first = beta1 * m[i] + (1 - beta1) * grad[i];
  float second = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
  m_out[i] = first;
  v_out[i] = second;
  float mhat = first / (1 - pow(beta1, float(step)));
  float vhat = second / (1 - pow(beta2, float(step)));
  params_out[i] = params[i] * (1 - lr * decay) - lr * mhat / (sqrt(vhat) + eps);
}
