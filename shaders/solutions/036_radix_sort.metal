// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const uint* keys [[buffer(0)]],
    device uint* sorted [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    device uint* scratch [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  device const uint* source = keys;
  for (uint pass = 0; pass < 4; ++pass) {
    device uint* destination = (pass % 2 == 0) ? scratch : sorted;
    uint offsets[256] = {};
    uint shift = 8 * pass;
    for (uint j = 0; j < n; ++j)
      ++offsets[(source[j] >> shift) & 255];
    uint start = 0;
    for (uint digit = 0; digit < 256; ++digit) {
      uint size = offsets[digit];
      offsets[digit] = start;
      start += size;
    }
    for (uint j = 0; j < n; ++j) {
      uint key = source[j], digit = (key >> shift) & 255;
      destination[offsets[digit]++] = key;
    }
    source = destination;
  }
}
