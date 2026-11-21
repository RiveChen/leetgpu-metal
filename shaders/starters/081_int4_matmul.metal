#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* x [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const half* scale [[buffer(2)]],
    device half* y [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply FP16 inputs by group-scaled INT4 weights. Even K indices use the high nibble.
}
