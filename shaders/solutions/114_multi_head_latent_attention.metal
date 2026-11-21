// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* cache [[buffer(1)]],
    device const float* uk [[buffer(2)]],
    device const float* uv [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& rope [[buffer(9)]],
    device float* scratch [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {

  threadgroup float query[512], value[512], peak[64], mass[64];
  device float* scores = scratch + group * length;
  for (uint r = lane; r < rank; r += 64) {
    float a = 0;
    for (uint d = 0; d < width; ++d)
      a += q[group * (width + rope) + d] * uk[(group * width + d) * rank + r];
    query[r] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float high = -INFINITY;
  for (uint j = lane; j < length; j += 64) {
    device const float* token = cache + j * (rank + rope);
    float s = 0;
    for (uint r = 0; r < rank; ++r)
      s += query[r] * token[r];
    for (uint d = 0; d < rope; ++d)
      s += q[group * (width + rope) + width + d] * token[rank + d];
    s *= rsqrt(float(width + rope));
    scores[j] = s;
    high = max(high, s);
  }
  peak[lane] = high;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < 64; ++i)
    high = max(high, peak[i]);
  float total = 0;
  for (uint j = lane; j < length; j += 64) {
    float w = exp(scores[j] - high);
    scores[j] = w;
    total += w;
  }
  mass[lane] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  total = 0;
  for (uint i = 0; i < 64; ++i)
    total += mass[i];
  for (uint r = lane; r < rank; r += 64) {
    float s = 0;
    for (uint j = 0; j < length; ++j)
      s += scores[j] * cache[j * (rank + rope) + r];
    value[r] = s / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = lane; d < width; d += 64) {
    float s = 0;
    for (uint r = 0; r < rank; ++r)
      s += value[r] * uv[(group * rank + r) * width + d];
    out[group * width + d] = s;
  }
}
