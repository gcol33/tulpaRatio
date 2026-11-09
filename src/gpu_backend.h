// gpu_backend.h
// GPU acceleration for GP computations
// Supports CUDA (NVIDIA) and OpenCL (cross-platform)
//
// Uses RUNTIME detection - works even if user installs CUDA/OpenCL after
// installing tulpaRatio. No recompilation needed.
//
// On Windows: looks for nvcuda.dll (CUDA) or OpenCL.dll
// On Linux: looks for libcuda.so or libOpenCL.so
// On macOS: looks for CUDA.framework or OpenCL.framework

#ifndef RATIOD_GPU_BACKEND_H
#define RATIOD_GPU_BACKEND_H

#include <string>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  #define GPU_LIB_HANDLE HMODULE
  #define GPU_LOAD_LIB(name) LoadLibraryA(name)
  #define GPU_GET_PROC(lib, name) GetProcAddress(lib, name)
  #define GPU_FREE_LIB(lib) FreeLibrary(lib)
  #define CUDA_LIB_NAME "nvcuda.dll"
  #define OPENCL_LIB_NAME "OpenCL.dll"
#else
  #include <dlfcn.h>
  #define GPU_LIB_HANDLE void*
  #define GPU_LOAD_LIB(name) dlopen(name, RTLD_LAZY)
  #define GPU_GET_PROC(lib, name) dlsym(lib, name)
  #define GPU_FREE_LIB(lib) dlclose(lib)
  #ifdef __APPLE__
    #define CUDA_LIB_NAME "/usr/local/cuda/lib/libcuda.dylib"
    #define OPENCL_LIB_NAME "/System/Library/Frameworks/OpenCL.framework/OpenCL"
  #else
    #define CUDA_LIB_NAME "libcuda.so.1"
    #define OPENCL_LIB_NAME "libOpenCL.so.1"
  #endif
#endif

namespace ratiod_gpu {

// =============================================================================
// GPU Device Information
// =============================================================================

struct GPUDeviceInfo {
  std::string name;
  size_t memory_mb;
  std::string compute_capability;  // CUDA only
  int device_id;
};

struct GPUInfo {
  bool available;
  std::string backend;  // "cuda", "opencl", or "none"
  int device_count;
  std::vector<GPUDeviceInfo> devices;
};

// =============================================================================
// Runtime GPU Detection
// =============================================================================

// Try to load CUDA driver library at runtime
inline bool try_load_cuda() {
  GPU_LIB_HANDLE lib = GPU_LOAD_LIB(CUDA_LIB_NAME);
  if (lib != nullptr) {
    GPU_FREE_LIB(lib);
    return true;
  }
  return false;
}

// Try to load OpenCL library at runtime
inline bool try_load_opencl() {
  GPU_LIB_HANDLE lib = GPU_LOAD_LIB(OPENCL_LIB_NAME);
  if (lib != nullptr) {
    GPU_FREE_LIB(lib);
    return true;
  }
  return false;
}

// Check GPU availability at runtime (no recompilation needed)
inline bool gpu_available() {
  static int cached = -1;  // Cache result: -1 = not checked, 0 = no, 1 = yes
  if (cached >= 0) {
    return cached == 1;
  }

  // Try CUDA first (usually faster), then OpenCL
  if (try_load_cuda() || try_load_opencl()) {
    cached = 1;
    return true;
  }

  cached = 0;
  return false;
}

// Get detailed GPU info
inline GPUInfo get_gpu_info() {
  GPUInfo info;
  info.available = false;
  info.backend = "none";
  info.device_count = 0;

  // Check CUDA
  if (try_load_cuda()) {
    info.available = true;
    info.backend = "cuda";
    // Note: To get device count/info, we'd need to actually call CUDA API
    // For now, just report that CUDA is available
    info.device_count = 1;  // Assume at least 1 if library loads

    GPUDeviceInfo dev;
    dev.name = "CUDA Device (details require CUDA initialization)";
    dev.memory_mb = 0;
    dev.compute_capability = "unknown";
    dev.device_id = 0;
    info.devices.push_back(dev);
    return info;
  }

  // Check OpenCL
  if (try_load_opencl()) {
    info.available = true;
    info.backend = "opencl";
    info.device_count = 1;  // Assume at least 1 if library loads

    GPUDeviceInfo dev;
    dev.name = "OpenCL Device (details require OpenCL initialization)";
    dev.memory_mb = 0;
    dev.compute_capability = "";
    dev.device_id = 0;
    info.devices.push_back(dev);
    return info;
  }

  return info;
}

}  // namespace ratiod_gpu

#endif  // RATIOD_GPU_BACKEND_H
