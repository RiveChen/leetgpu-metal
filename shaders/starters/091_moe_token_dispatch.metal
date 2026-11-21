#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* routes [[buffer(1)]],
    device float* dispatched [[buffer(2)]],
    device int* counts [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& experts [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Pack tokens by expert in input order, zero unused slots, and write per-expert counts.
}
