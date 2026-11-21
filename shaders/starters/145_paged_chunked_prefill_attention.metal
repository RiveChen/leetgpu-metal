#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* contexts [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& seqs [[buffer(7)]],
    constant uint& total [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& kv [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant uint& block [[buffer(12)]],
    constant uint& slots [[buffer(13)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Gather paged KV blocks and attend causally from the final query chunk of each sequence.
}
