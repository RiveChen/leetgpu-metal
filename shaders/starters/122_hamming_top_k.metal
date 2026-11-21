#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* codes [[buffer(0)]],
    device const int* query [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& top [[buffer(4)]],
    device ushort* distances [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]]
) {
  // TODO: Return k nearest 256-bit codes by Hamming distance, breaking ties by row index.
}
