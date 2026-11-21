// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float2 rotate(float2 value, uint phase, uint period) {
  float cosine, sine = sincos(-2 * M_PI_F * float(phase) / float(period), cosine);
  return float2(value.x * cosine - value.y * sine, value.x * sine + value.y * cosine);
}

kernel void solve(
    device const float2* signal [[buffer(0)]],
    device float2* spectrum [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  if (n == 1) {
    if (i == 0)
      spectrum[0] = signal[0];
    return;
  }
  if ((n & (n - 1)) != 0) {
    // Direct DFT for lengths that are not powers of two.
    for (uint k = i; k < n; k += 64) {
      float2 sum = 0;
      for (uint j = 0; j < n; ++j)
        sum += rotate(signal[j], uint((ulong(k) * j) % n), n);
      spectrum[k] = sum;
    }
    return;
  }
  uint levels = ctz(n);
  for (uint j = i; j < n; j += 64)
    spectrum[reverse_bits(j) >> (32 - levels)] = signal[j];
  for (uint span = 2; span <= n; span *= 2) {
    threadgroup_barrier(mem_flags::mem_device);
    uint half_span = span / 2;
    for (uint j = i; j < n / 2; j += 64) {
      uint phase = j % half_span, first = (j / half_span) * span + phase;
      float2 even = spectrum[first], odd = rotate(spectrum[first + half_span], phase, span);
      spectrum[first] = even + odd;
      spectrum[first + half_span] = even - odd;
    }
  }
}
