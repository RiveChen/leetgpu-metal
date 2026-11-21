#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* dist [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& nodes [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  // TODO: Run Floyd-Warshall, preserving +infinity for pairs without a path.
}
