#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* grid [[buffer(0)]],
    device int* result [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& start_row [[buffer(4)]],
    constant uint& start_col [[buffer(5)]],
    constant uint& end_row [[buffer(6)]],
    constant uint& end_col [[buffer(7)]],
    device int* scratch [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Find the four-neighbor shortest path length through zeros, or return -1. scratch holds 2*rows*cols integers for the queue and distances.
}
