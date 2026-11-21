#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& model_width [[buffer(5)]],
    constant uint& head_count [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply scaled dot-product self-attention independently to each head in the row-major [n,model_width] matrices.
}
