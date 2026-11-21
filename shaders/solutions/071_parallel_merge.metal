// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m + n)
    return;
  bool left = i < m;
  uint pos = left ? i : i - m;
  float value = left ? a[pos] : b[pos];
  device const float* other = left ? b : a;
  uint lo = 0, hi = left ? n : m;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (other[mid] < value || (!left && other[mid] == value))
      lo = mid + 1;
    else
      hi = mid;
  }
  c[pos + lo] = value;
}
