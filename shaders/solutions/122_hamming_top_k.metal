// AI-generated reference solution.
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
  // Distances between 256-bit vectors have only 257 possible values.
  threadgroup atomic_uint counts[257];
  for (uint d = lane; d <= 256; d += 64)
    atomic_store_explicit(counts + d, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = lane; row < rows; row += 64) {
    uint d = 0;
    for (uint word = 0; word < 8; ++word)
      d += popcount(as_type<uint>(codes[row * 8 + word] ^ query[word]));
    distances[row] = ushort(d);
    atomic_fetch_add_explicit(counts + d, 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (lane == 0) {
    uint offsets[257], prefix = 0;
    for (uint d = 0; d <= 256; ++d) {
      offsets[d] = prefix;
      prefix += atomic_load_explicit(counts + d, memory_order_relaxed);
    }
    // Scanning in row order preserves the lower-index tie break.
    uint written = 0;
    for (uint row = 0; row < rows && written < top; ++row) {
      uint rank = offsets[distances[row]]++;
      if (rank < top) {
        indices[rank] = int(row);
        ++written;
      }
    }
  }
}
