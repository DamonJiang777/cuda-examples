#ifndef COMMOM_HELPER_CUDA_DRVAPI_H_
#define COMMON_HELPER_CUDA_DRVAPI_H_

#include "cuda.h"
#include "helper_string.h"

#ifdef __cuda_cuda_h__
// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper fucntions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
  if (err != CUDA_SUCCESS)
  {
    const char *error_str = NULL;
    cuGetErrorString(err, &error_str);
    fprintf(stderr,
            "checkCudaErrors() Driver Api error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err,
            error_str,
            file,
            line);
    exit(EXIT_FAILURE);
  }
}
#endif
#endif

// This functions wraps teh CUDA Driver api into a template function
template<class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
  checkCudaErrors(cuDeviceGetAttribute(attribute, device_attribute, device));
}
// General GPU Device CUDA Initialization
inline int gpuDeviceInitDRV(int argc, const char **argv)
{
  int cu_device = 0;
  int device_cnt = 0;
  checkCudaErrors(cuInit(0));

  checkCudaErrors(cuDeviceGetCount(&device_cnt));

  if (device_cnt == 0)
  {
    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;
  dev = GetCmdLineArgumentInt(argc, (const char **) argv, "device=");

  if (dev < 0)
  {
    dev = 0;
  }

  if (dev > device_cnt - 1)
  {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_cnt);
    fprintf(stderr, ">> cudaDeviceInit (-device=%d) is not a valid GPU device.<<\n", dev);
    fprintf(stderr, "\n");
    return -dev;
  }

  checkCudaErrors(cuDeviceGet(&cu_device, dev));
  char name[100];
  checkCudaErrors(cuDeviceGetName(name, 100, cu_device));

  int compute_mode;
  getCudaAttribute<int>(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);

  if (compute_mode == CU_COMPUTEMODE_PROHIBITED)
  {
    fprintf(stderr,
            "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no"
            " threads can use this CUDA device.\n");
    return -1;
  }

  if (CheckCmdLineFlag(argc, (const char **) argv, "quite") == false)
  {
    printf("gpuDeviceInitDRV() using CUDA Device [%d]: %s\n", dev, name);
  }

  return dev;
}

inline int _ConvertSMVer2CoresDRV(int major, int minor)
{
  // defines for GPU architecure types (using the SM version to determine the #
  // of cores per sm)
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[]{{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                  {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                  {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                  {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xa1, 128},
                                  {0xc0, 128}, {-1, -1}};

  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // if we don't find the values, we default use the previous one to run
  // properly
  printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n",
         major,
         minor,
         nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores; // {0xc0, 128}
}
// This function returns the best GPU base on performance
inline int gpuGetMaxGflopsDeviceIdDRV()
{
  CUdevice current_device = 0;
  CUdevice max_perf_device = 0;
  int device_count = 0;
  int sm_per_multiproc = 0;
  unsigned long long max_compute_perf = 0;
  int major = 0;
  int minor = 0;
  int multi_process_count;
  int clock_rate;
  int device_prohibited = 0;

  cuInit(0);
  checkCudaErrors(cuDeviceGetCount(&device_count));

  if (device_count == 0)
  {
    fprintf(stderr, "gpuGetMaxGflopsDeviceIdDRV() error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  // find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count)
  {
    checkCudaErrors(cuDeviceGetAttribute(&multi_process_count,
                                         CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                         current_device));
    checkCudaErrors(
        cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, current_device));
    checkCudaErrors(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, current_device));
    checkCudaErrors(
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, current_device));

    int compute_mode;
    getCudaAttribute<int>(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, current_device);
    if (compute_mode != CU_COMPUTEMODE_PROHIBITED)
    {
      if (major == 9999 && minor == 9999)
      {
        // virtual device
        sm_per_multiproc = 1;
      }
      else
      {
        sm_per_multiproc = _ConvertSMVer2CoresDRV(major, minor);
      }

      unsigned long long computer_perf =
          ((unsigned long long) multi_process_count * sm_per_multiproc * clock_rate);

      if (computer_perf > max_compute_perf)
      {
        max_compute_perf = computer_perf;
        max_perf_device = current_device;
      }
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
            "gpuGetMaxGflopsDeviceIdDRV() error: all devices have compute mode "
            "prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}
// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDeviceDRV(int argc, const char **argv)
{
  CUdevice cu_device;
  int device_id = 0;

  // if the command-line has a device number specified, use it
  if (CheckCmdLineFlag(argc, (const char **) argv, "device"))
  {
    device_id = gpuDeviceInitDRV(argc, argv);

    if (device_id < 0)
    {
      printf("exiting...\n");
      exit(EXIT_SUCCESS); // TODO(jiangdz): success ?
    }
  }
  else
  {
    // otherwise pick the device with highest Gflops/s
    char name[100];
    device_id = gpuGetMaxGflopsDeviceIdDRV();
    checkCudaErrors(cuDeviceGet(&cu_device, device_id));
    cuDeviceGetName(name, 100, cu_device);
    printf("> Using CUDA device [%d]: %s\n", device_id, name);
  }

  cuDeviceGet(&cu_device, device_id);
  return cu_device;
}

#endif
