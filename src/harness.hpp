// src/harness.hpp: C ABI interface for LeetGPU in Metal
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct BufferDesc {
  void* ptr;          // must be 4096-byte page-aligned continuous memory pointer
  size_t byte_length; // byte length
  uint32_t index;     // Metal [[buffer(index)]] binding slot
};

struct BytesDesc {
  const void* ptr;    // scalar or structure constant pointer (e.g. uint N, M, K)
  size_t byte_length; // byte length (<= 4096 bytes)
  uint32_t index;     // Metal [[buffer(index)]] binding slot
};

struct DispatchConfig {
  uint32_t grid_x, grid_y, grid_z;
  uint32_t tg_x, tg_y, tg_z;
  bool use_threadgroups; // false = dispatchThreads (grid is total thread count)
                         // true  = dispatchThreadgroups (grid is threadgroup count)
};

struct RunResult {
  int status;           // 0 = success, -1 = compilation failed, -2 = runtime GPU error
  double median_ms;     // median hardware time (ms) of timed_runs executions
  double min_ms;        // minimum hardware time (ms)
  char error_msg[2048]; // error details
};

RunResult run_metal_kernel(
    const char* msl_source,
    const char* entry_name,
    const BufferDesc* buffers,
    size_t num_buffers,
    const BytesDesc* bytes_args,
    size_t num_bytes_args,
    DispatchConfig dispatch,
    int warmup_runs,
    int timed_runs,
    bool flush_l2,
    bool fast_math
);

// Restore the payload of buffer 0 before each dispatch, outside GPU timing.
RunResult run_metal_kernel_inplace(
    const char* msl_source,
    const char* entry_name,
    const BufferDesc* buffers,
    size_t num_buffers,
    const BytesDesc* bytes_args,
    size_t num_bytes_args,
    DispatchConfig dispatch,
    int warmup_runs,
    int timed_runs,
    bool flush_l2,
    bool fast_math,
    const void* initial_data,
    size_t initial_length
);

} // extern "C"
