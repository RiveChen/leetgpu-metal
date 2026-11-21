// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* trig [[buffer(2)]],
    device const int* positions [[buffer(3)]],
    device const float* old_keys [[buffer(4)]],
    device const float* old_values [[buffer(5)]],
    device float* query [[buffer(6)]],
    device float* keys [[buffer(7)]],
    device float* values [[buffer(8)]],
    constant uint& batch [[buffer(9)]],
    constant uint& model [[buffer(10)]],
    constant uint& q_heads [[buffer(11)]],
    constant uint& kv_heads [[buffer(12)]],
    constant uint& width [[buffer(13)]],
    constant uint& capacity [[buffer(14)]],
    uint i [[thread_position_in_grid]]
) {
  uint stride = (q_heads + 2 * kv_heads) * width;
  uint query_size = batch * q_heads * width;
  if (i < query_size) {
    uint b = i / (q_heads * width), d = i % width, j = i % (q_heads * width);
    uint pair = d < width / 2 ? j + width / 2 : j - width / 2;
    float a = 0, other = 0;
    for (uint k = 0; k < model; ++k) {
      float input = x[b * model + k];
      a += input * weights[k * stride + j];
      other += input * weights[k * stride + pair];
    }
    uint t = positions[b] * width + d % (width / 2);
    query[i] = a * trig[t] + (d < width / 2 ? -other : other) * trig[t + width / 2];
  }
  if (i >= batch * kv_heads * capacity * width)
    return;
  uint b = i / (kv_heads * capacity * width), head = (i / (capacity * width)) % kv_heads;
  uint position = (i / width) % capacity, d = i % width;
  if (position != uint(positions[b])) {
    keys[i] = old_keys[i];
    values[i] = old_values[i];
    return;
  }
  uint j = (q_heads + head) * width + d;
  uint pair = d < width / 2 ? j + width / 2 : j - width / 2;
  float a = 0, other = 0, value = 0;
  for (uint k = 0; k < model; ++k) {
    float input = x[b * model + k];
    a += input * weights[k * stride + j];
    other += input * weights[k * stride + pair];
    value += input * weights[k * stride + j + kv_heads * width];
  }
  uint t = position * width + d % (width / 2);
  keys[i] = a * trig[t] + (d < width / 2 ? -other : other) * trig[t + width / 2];
  values[i] = value;
}
