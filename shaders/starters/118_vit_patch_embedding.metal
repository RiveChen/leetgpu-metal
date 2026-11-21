#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* images [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* cls [[buffer(3)]],
    device const float* pos [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& height [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant uint& patch [[buffer(10)]],
    constant uint& embed [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Project nonoverlapping NCHW patches, prepend the class token and add positions.
}
