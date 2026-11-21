#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
  // TODO: Write the inclusive prefix sum of the N input elements.
}
