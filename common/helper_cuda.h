#ifndef COMMON_CUDA_HELPER_H_
#define COMMON_CUDA_HELPER_H_

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "helper_string.h"

static const char *_cudaGetErrorNum(cudaError_t error)
{
  return cudaGetErrorName(error);
}
template<typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    fprintf(stderr,
            "CUDA error at %s:%d, code=%s(%d) \"%s\"",
            file,
            line,
            static_cast<unsigned int>(result),
            _cudaGetErrorNum(result),
            func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline const char *_ConvertSMVer2ArchName(int major, int minor)
{
  typedef struct
  {
    int SM; // OxMm (hexidecimal notation), M = SM Major version,
    // m = SM minor version
    const char *name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},    {0x32, "Kepler"},       {0x35, "Kepler"},    {0x37, "Kepler"},
      {0x50, "Maxwell"},   {0x52, "Maxwell"},      {0x53, "Maxwell"},   {0x60, "Pascal"},
      {0x61, "Pascal"},    {0x62, "Pascal"},       {0x70, "Volta"},     {0x72, "Xavier"},
      {0x75, "Turing"},    {0x80, "Ampere"},       {0x86, "Ampere"},    {0x87, "Ampere"},
      {0x89, "Ada"},       {0x90, "Hopper"},       {0xa0, "Blackwell"}, {0xa1, "Blackwell"},
      {0xc0, "Blackwell"}, {-1, "Graphics Device"}};

  int index = 0;
  while (nGpuArchNameSM[index].SM != -1)
  {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchNameSM[index].name;
    }
    index++;
  }

  // if we don't find the values, we default use the previous one
  // to run properly
  printf("MapSMtoArchName for SM %d.%d is undefined."
         " Default to use %s\n",
         major,
         minor,
         nGpuArchNameSM[index - 1].name); // Graphics Device
  return nGpuArchNameSM[index - 1].name;
}

inline int _ConvertSMVer2Cores(int major, int minor)
{
  // Define for GPU Architecture types (using th SM version to determine
  // the # of cores per SM)
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCors;

  sSMtoCors nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                    {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                    {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                    {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xa1, 128},
                                    {0xc0, 129}, {-1, -1}};
  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  // if we don't find th values, we default use the previous one
  // to run properly
  printf("MapSMtoCores for SM %d.%d is undefined."
         " Default to use %d Cores/SM\n",
         major,
         minor,
         nGpuArchCoresPerSM[index - 1].Cores);

  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int GpuDeviceInit(int device_id)
{
  int device_cnt;
  checkCudaErrors(cudaGetDeviceCount(&device_cnt));

  if (device_cnt == 0)
  {
    fprintf(stderr,
            "GpuDeviceInit CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (device_id < 0)
  {
    device_id = 0;
  }

  if (device_id > device_cnt - 1)
  {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_cnt);
    fprintf(stderr,
            ">> GpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            device_id);
    fprintf(stderr, "\n");
    return -device_id;
  }

  int compute_mode = -1, major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, device_id));
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));

  if (compute_mode == cudaComputeModeProhibited)
  {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            " Prohibited>, no threads can use cudaSetDevice()\n");

    return -1;
  }

  if (major < 1)
  {
    fprintf(stderr, "GPUDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(device_id));
  printf("GPUDeviceInit CUDA Device [%d]: \"%s\"\n",
         device_id,
         _ConvertSMVer2ArchName(major, minor));

  return device_id;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int GpuGetMaxGflopsDeviceId()
{
  int current_device = 0;
  int sm_per_multiproc = 0;
  int max_pref_device = 0;
  int device_count = 0;
  int device_prohibited = 0;

  uint64_t max_compute_pref = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0)
  {
    fprintf(stderr,
            "GpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  while (current_device < device_count)
  {
    int compute_mode = -1;
    int major = 0;
    int minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    // if this GPU is not running on Compute Mode prohibited
    // the we can add it to the list
    if (compute_mode != cudaComputeModeProhibited)
    {
      if (major == 9999 and minor == 9999)
      {
        sm_per_multiproc = 1;
      }
      else
      {
        sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
      }
    }

    int multi_process_cnt = 0;
    int clock_rate = 0;
    checkCudaErrors(
        cudaDeviceGetAttribute(&multi_process_cnt, cudaDevAttrMultiProcessorCount, current_device));
    cudaError_t result = cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, current_device);

    if (result != cudaSuccess)
    {
      // if cudaDevAttrClockRate is not supported, we
      // set clock_rate as 1, to consider GPU with the most SMs adn CUDA Cores
      if (result == cudaErrorInvalidValue)
      {
        clock_rate = 1;
      }
      else
      {
        fprintf(stderr,
                "CUDA error at %s:%d, code=%d(%s)\n",
                __FILE__,
                __LINE__,
                static_cast<unsigned int>(result),
                _cudaGetErrorNum(result));
        exit(EXIT_FAILURE);
      }
    }
    uint64_t compute_perf = (uint64_t) multi_process_cnt * sm_per_multiproc * clock_rate;

    if (compute_perf > max_compute_pref)
    {
      max_compute_pref = compute_perf;
      max_pref_device = current_device;
    }
    else
    {
      device_prohibited++;
    }
    current_device++;
  }

  if (device_prohibited == device_count)
  {
    fprintf(stderr,
            "GpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute prohibited.\n");
    exit(EXIT_FAILURE);
  }
  return max_pref_device;
}

// Initialization code to find the best cuda device
inline int FindCudaDevice(int argc, const char *argv[])
{
  int device_id = 0;

  // if the command-line has a device number specified, use it
  if (CheckCmdLineFlag(argc, argv, "device"))
  {
    device_id = GetCmdLineArgumentInt(argc, argv, "device=");

    if (device_id < 0)
    {
      printf("Invalid command line parameter!\n");
      exit(EXIT_FAILURE);
    }
    else
    {
      device_id = GpuDeviceInit(device_id);
      if (device_id < 0)
      {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  else
  {
    // pick the device with highest Gflops/s
    device_id = GpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(device_id));
    int major = 0;
    int minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           device_id,
           _ConvertSMVer2ArchName(major, minor),
           major,
           minor);
  }
  return device_id;
}
#endif
