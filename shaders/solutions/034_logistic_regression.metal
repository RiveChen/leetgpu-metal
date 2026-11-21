// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Factor a symmetric positive-definite matrix and solve for rhs in place.
inline void solve_spd(device float* matrix, device float* rhs, uint n) {
  for (uint row = 0; row < n; ++row) {
    for (uint col = 0; col <= row; ++col) {
      float value = matrix[row * n + col];
      for (uint j = 0; j < col; ++j)
        value -= matrix[row * n + j] * matrix[col * n + j];
      matrix[row * n + col] = row == col ? sqrt(value) : value / matrix[col * n + col];
    }
    float value = rhs[row];
    for (uint j = 0; j < row; ++j)
      value -= matrix[row * n + j] * rhs[j];
    rhs[row] = value / matrix[row * n + row];
  }
  for (int row = int(n) - 1; row >= 0; --row) {
    float value = rhs[row];
    for (uint j = uint(row) + 1; j < n; ++j)
      value -= matrix[j * n + uint(row)] * rhs[j];
    rhs[row] = value / matrix[uint(row) * n + uint(row)];
  }
}

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* beta [[buffer(2)]],
    constant uint& samples [[buffer(3)]],
    constant uint& features [[buffer(4)]],
    device float* workspace [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  device float* gradient = workspace + features * features;
  for (uint j = 0; j < features; ++j)
    beta[j] = 0;
  for (uint iter = 0; iter < 100; ++iter) {
    for (uint j = 0; j < features; ++j) {
      gradient[j] = 1e-6f * beta[j];
      for (uint k = 0; k <= j; ++k)
        workspace[j * features + k] = (j == k) ? 1e-6f : 0.0f;
    }
    for (uint row = 0; row < samples; ++row) {
      device const float* sample = x + row * features;
      float logit = 0;
      for (uint j = 0; j < features; ++j)
        logit += sample[j] * beta[j];
      float p = 1 / (1 + exp(-logit)), weight = max(p * (1 - p), 1e-8f);
      for (uint j = 0; j < features; ++j) {
        gradient[j] += sample[j] * (p - y[row]);
        for (uint k = 0; k <= j; ++k)
          workspace[j * features + k] += weight * sample[j] * sample[k];
      }
    }
    solve_spd(workspace, gradient, features);
    float step = 0;
    for (uint j = 0; j < features; ++j) {
      beta[j] -= gradient[j];
      step += gradient[j] * gradient[j];
    }
    if (step < 1e-12f)
      break;
  }
}
