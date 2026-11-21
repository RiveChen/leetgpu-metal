// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* src [[buffer(0)]],
    device const int* dst [[buffer(1)]],
    device int* labels [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& edges [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  for (uint j = 0; j < n; ++j)
    labels[j] = int(j);
  for (uint j = 0; j < edges; ++j) {
    int a = src[j], b = dst[j];
    while (a != labels[a])
      a = labels[a];
    while (b != labels[b])
      b = labels[b];
    labels[max(a, b)] = min(a, b);
  }
  for (uint j = 0; j < n; ++j) {
    int root = labels[j];
    while (root != labels[root])
      root = labels[root];
    labels[j] = root;
  }
}
