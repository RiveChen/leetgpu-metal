#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device const int* ids [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& features [[buffer(7)]],
    constant uint& outputs [[buffer(8)]],
    constant uint& rank [[buffer(9)]],
    constant uint& adapters [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Project tokens with the base matrix and the selected LoRA adapter; -1 selects only the base.
}
