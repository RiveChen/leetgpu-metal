#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* prompts [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // w[0..1] parameterize digit embeddings, w[2..4] the query/value projections,
  // w[5..7] the gated MLP, and w[8..9] the final RMSNorm scales.
  // TODO: Decode eleven greedy digit steps for each encoded addition prompt. Write all ten logits per step.
}
