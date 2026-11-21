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
  // TODO: Label each vertex with the smallest vertex index in its undirected connected component.
}
