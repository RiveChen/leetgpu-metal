// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* grad [[buffer(0)]],
    device const float* buf [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& momentum [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  uint h = min(rows, cols), w = max(rows, cols), size = h * w;
  bool tall = rows > cols;
  device float* state = scratch;
  device float* gram = state + size;
  device float* correction = gram + h * h;
  // A single threadgroup synchronizes all five matrix iterations.
  threadgroup float partial[64];
  float norm2 = 0;
  for (uint j = i; j < size; j += 64) {
    uint src = tall ? (j % w) * cols + j / w : j;
    float v = momentum * buf[src] + grad[src];
    state[j] = v;
    norm2 += v * v;
  }
  partial[i] = norm2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 32; step; step /= 2) {
    if (i < step)
      partial[i] += partial[i + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = i; j < size; j += 64)
    state[j] /= (sqrt(partial[0]) + 1e-7f);
  threadgroup_barrier(mem_flags::mem_device);
  for (uint iteration = 0; iteration < 5; ++iteration) {
    for (uint j = i; j < h * h; j += 64) {
      uint r = j / h, c = j % h;
      float sum = 0;
      for (uint k = 0; k < w; ++k)
        sum += state[r * w + k] * state[c * w + k];
      gram[j] = sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = i; j < h * h; j += 64) {
      uint r = j / h, c = j % h;
      float sum = 0;
      for (uint k = 0; k < h; ++k)
        sum += gram[r * h + k] * gram[k * h + c];
      correction[j] = -4.775f * gram[j] + 2.0315f * sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = i; j < size; j += 64) {
      uint r = j / w, c = j % w;
      float sum = 0;
      for (uint k = 0; k < h; ++k)
        sum += correction[r * h + k] * state[k * w + c];
      output[tall ? c * cols + r : j] = 3.4445f * state[j] + sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = i; j < size; j += 64)
      state[j] = output[tall ? (j % w) * cols + j / w : j];
    threadgroup_barrier(mem_flags::mem_device);
  }
  for (uint j = i; j < size; j += 64)
    output[j] *= sqrt(float(w));
}
