"""46. BFS Shortest Path."""

from collections import deque

import numpy as np

DTYPE = np.int32
RTOL, ATOL = 0, 0
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"rows": 1, "cols": 1, "start": (0, 0), "end": (0, 0), "wall": False},
    {"rows": 7, "cols": 11, "start": (2, 1), "end": (1, 9), "wall": True},
    {"rows": 23, "cols": 31, "start": (19, 2), "end": (3, 27), "wall": False},
]


def make_inputs(case, rng):
    rows, cols = case["rows"], case["cols"]
    start, end = case["start"], case["end"]
    grid = (rng.random((rows, cols)) < 0.2).astype(np.int32)
    if rows > 1:
        mid = cols // 2
        grid[:, mid] = 1
        if not case["wall"]:
            # Force a detour to the bottom row to pass the wall.
            grid[start[0] :, start[1]] = 0
            grid[-1, start[1] : end[1] + 1] = 0
            grid[end[0] :, end[1]] = 0
    grid[start] = grid[end] = 0
    return [grid], (1,), [np.uint32(v) for v in (rows, cols, *start, *end)]


def reference_impl(grid, rows, cols, sr, sc, er, ec):
    start, end = (int(sr), int(sc)), (int(er), int(ec))
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), steps = queue.popleft()
        if (r, c) == end:
            return np.array([steps], np.int32)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), steps + 1))
    return np.array([-1], np.int32)


def dispatch_config(case):
    return 1, 1, 1, 1, 1, 1, False


def workspace_bytes(case):
    return 8 * case["rows"] * case["cols"]
