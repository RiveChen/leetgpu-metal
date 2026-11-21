// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* values [[buffer(0)]],
    device const int2* shapes [[buffer(1)]],
    device const float2* locations [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& queries [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& levels [[buffer(8)]],
    constant uint& points [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= queries * heads * width)
    return;
  uint row = i / width, head = row % heads, d = i % width, start = 0;
  float sum = 0;
  for (uint l = 0; l < levels; ++l) {
    int2 shape = shapes[l];
    for (uint p = 0; p < points; ++p) {
      uint idx = (row * levels + l) * points + p;
      float2 pos = locations[idx] * float2(shape.y, shape.x) - 0.5f;
      int2 origin = int2(floor(pos));
      float2 fraction = pos - float2(origin);
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          int x = origin.x + dx, y = origin.y + dy;
          if (x < 0 || y < 0 || x >= shape.y || y >= shape.x)
            continue;
          float coeff = (dx ? fraction.x : 1 - fraction.x) * (dy ? fraction.y : 1 - fraction.y);
          sum +=
              weights[idx] * coeff * values[((start + y * shape.y + x) * heads + head) * width + d];
        }
    }
    start += shape.x * shape.y;
  }
  output[i] = sum;
}
