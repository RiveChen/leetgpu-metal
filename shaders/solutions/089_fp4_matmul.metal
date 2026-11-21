// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

constant float magnitudes[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
inline float fp4(uchar code) {
  return (code & 8 ? -1.0f : 1.0f) * magnitudes[code & 7];
}
inline float scale8(uchar code) {
  int e = (code >> 3) & 15;
  float m = float(code & 7) * 0.125f;
  return (code & 128 ? -1.0f : 1.0f) * ldexp((e != 0 ? 1.0f : 0.0f) + m, max(e, 1) - 7);
}
kernel void solve(
    device const uchar* x [[buffer(0)]],
    device const uchar* xs [[buffer(1)]],
    device const uchar* w [[buffer(2)]],
    device const uchar* ws [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& inner [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {

  uint row = i / cols, col = i % cols;
  float total = 0;
  for (uint block = 0; block < inner / 16; ++block) {
    float partial = 0;
    for (uint j = 0; j < 8; ++j) {
      uchar a = x[row * (inner / 2) + block * 8 + j], b = w[col * (inner / 2) + block * 8 + j];
      partial += fp4(a >> 4) * fp4(b >> 4) + fp4(a & 15) * fp4(b & 15);
    }
    total +=
        partial * scale8(xs[row * (inner / 16) + block]) * scale8(ws[col * (inner / 16) + block]);
  }
  out[i] = half(alpha * total);
}
