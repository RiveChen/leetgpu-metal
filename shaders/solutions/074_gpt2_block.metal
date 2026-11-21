// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

constant uint O_LN1 = 0;
constant uint O_LB1 = 768;
constant uint O_QKV = 1536;
constant uint O_BQKV = 1771008;
constant uint O_ATT = 1773312;
constant uint O_BATT = 2363136;
constant uint O_LN2 = 2363904;
constant uint O_LB2 = 2364672;
constant uint O_FC = 2365440;
constant uint O_BFC = 4724736;
constant uint O_PROJ = 4727808;
constant uint O_BPROJ = 7087104;
inline void project(
    device const float* x,
    device const float* w,
    device const float* b,
    device float* dst,
    uint length,
    uint inputs,
    uint outputs,
    uint lane
) {
  for (uint idx = lane; idx < length * outputs; idx += 128) {
    uint row = idx / outputs, col = idx % outputs;
    float sum = b[col];
    for (uint j = 0; j < inputs; ++j)
      sum += x[row * inputs + j] * w[j * outputs + col];
    dst[idx] = sum;
  }
  threadgroup_barrier(mem_flags::mem_device);
}

constant uint WIDTH = 768;
inline void normalize(
    device const float* x,
    device const float* gain,
    device const float* bias,
    device float* dst,
    uint length,
    uint lane
) {
  for (uint row = lane; row < length; row += 128) {
    device const float* src = x + row * WIDTH;
    float mean = 0;
    for (uint j = 0; j < WIDTH; ++j)
      mean += src[j];
    mean /= float(WIDTH);
    float energy = 0;
    for (uint j = 0; j < WIDTH; ++j) {
      float delta = src[j] - mean;
      energy += delta * delta;
    }
    float inv = rsqrt(energy / float(WIDTH) + 1e-5f);
    for (uint j = 0; j < WIDTH; ++j)
      dst[row * WIDTH + j] = (src[j] - mean) * inv * gain[j] + bias[j];
  }
  threadgroup_barrier(mem_flags::mem_device);
}
inline void attend(
    device const float* q,
    device const float* k,
    device const float* v,
    device float* context,
    uint length,
    uint lane
) {
  for (uint idx = lane; idx < length * WIDTH; idx += 128) {
    uint row = idx / WIDTH, head = idx % WIDTH / 64, col = idx % 64;
    device const float* query = q + row * 2304 + head * 64;
    float high = -INFINITY, mass = 0, a = 0;
    for (uint key = 0; key < length; ++key) {
      uint base = key * 2304 + (head / 1) * 64;
      float dot = 0;
      for (uint d = 0; d < 64; ++d)
        dot += query[d] * k[base + d];
      float score = dot * 0.125f, next = max(high, score), old = exp(high - next),
            w = exp(score - next);
      a = a * old + w * v[base + col];
      mass = mass * old + w;
      high = next;
    }
    context[idx] = a / mass;
  }
  threadgroup_barrier(mem_flags::mem_device);
}

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    device float* scratch [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  device float* norm = scratch;
  device float* qkv = norm + seq_len * 768;
  device float* context = qkv + seq_len * 2304;
  device float* fc = context + seq_len * 768;

  normalize(x, weights + O_LN1, weights + O_LB1, norm, seq_len, lane);
  project(norm, weights + O_QKV, weights + O_BQKV, qkv, seq_len, 768, 2304, lane);
  attend(qkv, qkv + 768, qkv + 1536, context, seq_len, lane);
  project(context, weights + O_ATT, weights + O_BATT, out, seq_len, 768, 768, lane);
  for (uint i = lane; i < seq_len * 768; i += 128)
    out[i] += x[i];
  threadgroup_barrier(mem_flags::mem_device);
  normalize(out, weights + O_LN2, weights + O_LB2, norm, seq_len, lane);
  project(norm, weights + O_FC, weights + O_BFC, fc, seq_len, 768, 3072, lane);
  for (uint i = lane; i < seq_len * 3072; i += 128) {
    float z = fc[i];
    fc[i] = 0.5f * z * (1.0f + tanh(0.7978845608f * (z + 0.044715f * z * z * z)));
  }
  threadgroup_barrier(mem_flags::mem_device);
  project(fc, weights + O_PROJ, weights + O_BPROJ, context, seq_len, 3072, 768, lane);
  for (uint i = lane; i < seq_len * 768; i += 128)
    out[i] += context[i];
}
