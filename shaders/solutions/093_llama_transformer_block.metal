// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

constant uint O_RMS1 = 0;
constant uint O_Q = 512;
constant uint O_K = 262656;
constant uint O_V = 328192;
constant uint O_O = 393728;
constant uint O_RMS2 = 655872;
constant uint O_GATE = 656384;
constant uint O_UP = 1377280;
constant uint O_DOWN = 2098176;
inline void project(
    device const float* x,
    device const float* w,
    device float* dst,
    uint length,
    uint inputs,
    uint outputs,
    uint lane
) {
  for (uint idx = lane; idx < length * outputs; idx += 128) {
    uint row = idx / outputs, col = idx % outputs;
    float sum = 0;
    for (uint j = 0; j < inputs; ++j)
      sum += x[row * inputs + j] * w[col * inputs + j];
    dst[idx] = sum;
  }
  threadgroup_barrier(mem_flags::mem_device);
}

constant uint WIDTH = 512;
inline void normalize(
    device const float* x, device const float* gain, device float* dst, uint length, uint lane
) {
  for (uint row = lane; row < length; row += 128) {
    device const float* src = x + row * WIDTH;
    float energy = 0;
    for (uint j = 0; j < WIDTH; ++j) {
      energy += src[j] * src[j];
    }
    float inv = rsqrt(energy / float(WIDTH) + 1e-5f);
    for (uint j = 0; j < WIDTH; ++j)
      dst[row * WIDTH + j] = src[j] * inv * gain[j];
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
    device const float* query = q + row * 512 + head * 64;
    float high = -INFINITY, mass = 0, a = 0;
    for (uint key = 0; key < row + 1; ++key) {
      uint base = key * 128 + (head / 4) * 64;
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

inline void rotate(
    device float* data,
    device const float* c,
    device const float* s,
    uint heads,
    uint length,
    uint lane
) {
  for (uint i = lane; i < length * heads * 32; i += 128) {
    uint offset = (i / 32) * 64 + i % 32, phase = i / (heads * 32) * 32 + i % 32;
    float a = data[offset], b = data[offset + 32];
    data[offset] = a * c[phase] - b * s[phase];
    data[offset + 32] = a * s[phase] + b * c[phase];
  }
  threadgroup_barrier(mem_flags::mem_device);
}
kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* cosine [[buffer(2)]],
    device const float* sine [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  device float* norm = scratch;
  device float* q = norm + seq_len * 512;
  device float* k = q + seq_len * 512;
  device float* v = k + seq_len * 128;
  device float* context = v + seq_len * 128;
  device float* gate = context + seq_len * 512;
  device float* up = gate + seq_len * 1408;

  normalize(x, weights + O_RMS1, norm, seq_len, lane);
  project(norm, weights + O_Q, q, seq_len, 512, 512, lane);
  project(norm, weights + O_K, k, seq_len, 512, 128, lane);
  project(norm, weights + O_V, v, seq_len, 512, 128, lane);
  rotate(q, cosine, sine, 8, seq_len, lane);
  rotate(k, cosine, sine, 2, seq_len, lane);
  attend(q, k, v, context, seq_len, lane);
  project(context, weights + O_O, out, seq_len, 512, 512, lane);
  for (uint i = lane; i < seq_len * 512; i += 128)
    out[i] += x[i];
  threadgroup_barrier(mem_flags::mem_device);
  normalize(out, weights + O_RMS2, norm, seq_len, lane);
  project(norm, weights + O_GATE, gate, seq_len, 512, 1408, lane);
  project(norm, weights + O_UP, up, seq_len, 512, 1408, lane);
  for (uint i = lane; i < seq_len * 1408; i += 128) {
    float z = gate[i];
    gate[i] = up[i] * z / (1.0f + exp(-z));
  }
  threadgroup_barrier(mem_flags::mem_device);
  project(gate, weights + O_DOWN, context, seq_len, 1408, 512, lane);
  for (uint i = lane; i < seq_len * 512; i += 128)
    out[i] += context[i];
}
