#include "harness.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

namespace {
// helper functions

constexpr size_t kFlushBytes = 64 * 1024 * 1024; // 64MB, covers SLC/L2 cache
constexpr const char* kFlushShader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void flush_cache(device uint* buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    buf[gid] = gid;
}
)";

void set_error(RunResult& res, int code, const char* msg) {
  res.status = code;
  snprintf(res.error_msg, sizeof(res.error_msg), "%s", msg ? msg : "Unknown error");
}

} // namespace

extern "C" RunResult run_metal_kernel_inplace(
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
) {
  RunResult res{};
  if (warmup_runs < 0 || timed_runs < 1) {
    set_error(res, -2, "warmup_runs must be >= 0 and timed_runs must be >= 1");
    return res;
  }
  if (initial_length && (!initial_data || !num_buffers || !buffers || !buffers[0].ptr ||
                         initial_length > buffers[0].byte_length)) {
    set_error(res, -2, "Invalid initial payload for in-place dispatch");
    return res;
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    NS::Array* dev_list = MTL::CopyAllDevices();
    if (dev_list && dev_list->count() > 0) {
      device = static_cast<MTL::Device*>(dev_list->object(0))->retain();
    }
    if (dev_list) {
      dev_list->release();
    }
  }
  if (!device) {
    set_error(res, -2, "Failed to create MTLDevice");
    pool->release();
    return res;
  }

  MTL::CommandQueue* queue = device->newCommandQueue();

  // 1. compile user Shader at runtime (not timed)
  NS::Error* error = nullptr;
  auto src_str = NS::String::string(msl_source, NS::UTF8StringEncoding);
  auto opts = MTL::CompileOptions::alloc()->init();
  opts->setMathMode(fast_math ? MTL::MathModeFast : MTL::MathModeSafe);

  MTL::Library* lib = device->newLibrary(src_str, opts, &error);
  opts->release();
  if (!lib) {
    set_error(
        res, -1, error ? error->localizedDescription()->utf8String() : "Shader compile failed"
    );
    queue->release();
    device->release();
    pool->release();
    return res;
  }

  auto fn_str = NS::String::string(entry_name, NS::UTF8StringEncoding);
  MTL::Function* fn = lib->newFunction(fn_str);
  if (!fn) {
    set_error(res, -1, "Entry function not found in Metal library");
    lib->release();
    queue->release();
    device->release();
    pool->release();
    return res;
  }

  MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);
  fn->release();
  lib->release();
  if (!pso) {
    set_error(res, -1, error ? error->localizedDescription()->utf8String() : "PSO creation failed");
    queue->release();
    device->release();
    pool->release();
    return res;
  }

  // 2. prepare L2/SLC flush pipeline and buffer
  MTL::ComputePipelineState* flush_pso = nullptr;
  MTL::Buffer* flush_buf = nullptr;
  if (flush_l2) {
    auto flush_src = NS::String::string(kFlushShader, NS::UTF8StringEncoding);
    MTL::Library* flush_lib = device->newLibrary(flush_src, nullptr, &error);
    MTL::Function* flush_fn =
        flush_lib->newFunction(NS::String::string("flush_cache", NS::UTF8StringEncoding));
    flush_pso = device->newComputePipelineState(flush_fn, &error);
    flush_buf = device->newBuffer(kFlushBytes, MTL::ResourceStorageModePrivate);
    flush_fn->release();
    flush_lib->release();
  }

  // 3. zero-copy wrap input/output UMA Buffer
  std::vector<MTL::Buffer*> mtl_buffers;
  mtl_buffers.reserve(num_buffers);
  for (size_t i = 0; i < num_buffers; ++i) {
    MTL::Buffer* b = device->newBuffer(
        buffers[i].ptr, buffers[i].byte_length, MTL::ResourceStorageModeShared, nullptr
    );
    if (!b) {
      set_error(res, -2, "newBufferWithBytesNoCopy failed (check 4096-byte page alignment)");
      for (auto* ptr : mtl_buffers)
        ptr->release();
      if (flush_buf)
        flush_buf->release();
      if (flush_pso)
        flush_pso->release();
      pso->release();
      queue->release();
      device->release();
      pool->release();
      return res;
    }
    mtl_buffers.push_back(b);
  }

  auto dispatch_once = [&](bool record_time, double& out_ms) -> bool {
    if (initial_length) {
      std::memcpy(buffers[0].ptr, initial_data, initial_length);
    }
    if (flush_l2 && flush_pso && flush_buf) {
      MTL::CommandBuffer* fcmd = queue->commandBuffer();
      MTL::ComputeCommandEncoder* fenc = fcmd->computeCommandEncoder();
      fenc->setComputePipelineState(flush_pso);
      fenc->setBuffer(flush_buf, 0, 0);
      uint32_t elems = static_cast<uint32_t>(kFlushBytes / sizeof(uint32_t));
      fenc->dispatchThreads(MTL::Size::Make(elems, 1, 1), MTL::Size::Make(256, 1, 1));
      fenc->endEncoding();
      fcmd->commit();
      fcmd->waitUntilCompleted();
    }

    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    for (size_t i = 0; i < num_buffers; ++i) {
      enc->setBuffer(mtl_buffers[i], 0, buffers[i].index);
    }
    for (size_t i = 0; i < num_bytes_args; ++i) {
      enc->setBytes(bytes_args[i].ptr, bytes_args[i].byte_length, bytes_args[i].index);
    }

    MTL::Size grid = MTL::Size::Make(dispatch.grid_x, dispatch.grid_y, dispatch.grid_z);
    MTL::Size tg = MTL::Size::Make(dispatch.tg_x, dispatch.tg_y, dispatch.tg_z);

    uint64_t tg_threads = static_cast<uint64_t>(dispatch.tg_x) * dispatch.tg_y * dispatch.tg_z;
    if (tg_threads > pso->maxTotalThreadsPerThreadgroup()) {
      enc->endEncoding();
      set_error(res, -2, "Threadgroup size exceeds pipeline maxTotalThreadsPerThreadgroup");
      return false;
    }

    if (dispatch.use_threadgroups) {
      enc->dispatchThreadgroups(grid, tg);
    } else {
      enc->dispatchThreads(grid, tg);
    }
    enc->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();

    if (cmd->status() != MTL::CommandBufferStatusCompleted) {
      NS::Error* cmd_err = cmd->error();
      set_error(
          res, -2,
          cmd_err ? cmd_err->localizedDescription()->utf8String() : "CommandBuffer execution failed"
      );
      return false;
    }

    if (record_time) {
      out_ms = (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
    }
    return true;
  };

  // 4. Warmup
  double dummy_ms = 0.0;
  for (int i = 0; i < warmup_runs; ++i) {
    if (!dispatch_once(false, dummy_ms))
      goto cleanup;
  }

  // 5. Timed Runs (default 15 times)
  {
    std::vector<double> samples;
    samples.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
      double ms = 0.0;
      if (!dispatch_once(true, ms))
        goto cleanup;
      samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    res.status = 0;
    res.min_ms = samples.front();
    res.median_ms = samples[samples.size() / 2];
  }

  // 6. Cleanup
cleanup:
  for (auto* b : mtl_buffers)
    b->release();
  if (flush_buf)
    flush_buf->release();
  if (flush_pso)
    flush_pso->release();
  pso->release();
  queue->release();
  device->release();
  pool->release();
  return res;
}

extern "C" RunResult run_metal_kernel(
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
) {
  return run_metal_kernel_inplace(
      msl_source, entry_name, buffers, num_buffers, bytes_args, num_bytes_args, dispatch,
      warmup_runs, timed_runs, flush_l2, fast_math, nullptr, 0
  );
}
