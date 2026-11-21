// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* grid [[buffer(0)]],
    device int* result [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& start_row [[buffer(4)]],
    constant uint& start_col [[buffer(5)]],
    constant uint& end_row [[buffer(6)]],
    constant uint& end_col [[buffer(7)]],
    device int* scratch [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  uint size = rows * cols, start = start_row * cols + start_col, end = end_row * cols + end_col;
  device int* queue = scratch;
  device int* distance = scratch + size;
  for (uint j = 0; j < size; ++j)
    distance[j] = -1;
  distance[start] = 0;
  queue[0] = int(start);
  uint head = 0, tail = 1;
  while (head < tail) {
    uint current = uint(queue[head++]);
    if (current == end)
      break;
    int2 cell = int2(current % cols, current / cols);
    const int2 directions[4] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
    for (uint j = 0; j < 4; ++j) {
      int2 neighbor = cell + directions[j];
      if (any(neighbor < 0) || neighbor.x >= int(cols) || neighbor.y >= int(rows))
        continue;
      uint next = uint(neighbor.y) * cols + uint(neighbor.x);
      if (grid[next] == 0 && distance[next] < 0) {
        distance[next] = distance[current] + 1;
        queue[tail++] = int(next);
      }
    }
  }
  result[0] = distance[end];
}
