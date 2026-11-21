// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline void sift_down(device float* values, uint root, uint end) {
  while (root < end / 2) {
    uint child = 2 * root + 1;
    if (child + 1 < end && values[child + 1] > values[child])
      ++child;
    if (values[root] >= values[child])
      break;
    float tmp = values[root];
    values[root] = values[child];
    values[child] = tmp;
    root = child;
  }
}

kernel void solve(
    device float* values [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
  for (uint root = n / 2; root > 0; --root)
    sift_down(values, root - 1, n);
  for (uint end = n; end > 1; --end) {
    float largest = values[0];
    values[0] = values[end - 1];
    values[end - 1] = largest;
    sift_down(values, 0, end - 1);
  }
}
