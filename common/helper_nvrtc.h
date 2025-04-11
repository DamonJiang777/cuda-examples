#ifndef COMMON_HELPER_NVRTC_H_
#define COMMON_HELPER_NVRTC_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "helper_cuda_drvapi.h"
#include "nvrtc.h"

#define NVRTC_SAFE_CALL(Name, x)                                                                   \
  do                                                                                               \
  {                                                                                                \
    nvrtcResult result = x;                                                                        \
    if (result != NVRTC_SUCCESS)                                                                   \
    {                                                                                              \
      std::cerr << "\nerror: " << Name << " failed with error " << nvrtcGetErrorString(result);    \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

void compileFileToCUBIN(char *filename,
                        int argc,
                        char **argv,
                        char **cubin_result,
                        size_t *cubin_result_size,
                        int requires_CGheaders)
{
  if (!filename)
  {
    std::cerr << "\nerrorL filename is empty for compileFileToCUBIN()\n";
    exit(1);
  }

  std::ifstream input_file(filename, std::ios::in | std::ios::binary | std::ios::ate);

  if (!input_file.is_open())
  {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = input_file.tellg();
  size_t input_size = (size_t) pos;
  char *mem_block = new char[input_size + 1];

  input_file.seekg(0, std::ios::beg);
  input_file.read(mem_block, input_size);
  input_file.close();
  mem_block[input_size] = '\x0';

  int num_compile_options = 0;

  char *compile_params[2];

  int major = 0, minor = 0;
  char device_name[256];

  // pick the best CUDA device avaiable
  CUdevice cu_device = findCudaDeviceDRV(argc, (const char **) argv);

  // get compute capabilities and the device name
  checkCudaErrors(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
  checkCudaErrors(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));

  {
    // compile cubin for the GPU arch on which are going to run cuda kernel
    std::string compile_options;
    compile_options = "--gpu-architecture=sm_";

    compile_params[num_compile_options] =
        reinterpret_cast<char *>(malloc(sizeof(char) * (compile_options.length() + 10)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    // only support linux
#else
    snprintf(compile_params[num_compile_options],
             compile_options.size() + 10,
             "%s%d%d",
             compile_options.c_str(),
             major,
             minor);
#endif
  }

  num_compile_options++;
  if (requires_CGheaders)
  {
    std::string compile_options;
    char header_names[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// only support linux
#else
    snprintf(header_names, sizeof(header_names), "%s", "cooperative_groups.h");
#endif

    compile_options = "--include-path=";
    char *str_path = sdkFindFilePath(header_names, argv[0]);
    if (!str_path)
    {
      std::cerr << "\error: header file " << header_names << " not found!\n";
      exit(1);
    }

    std::string path = str_path;
    if (!path.empty())
    {
      std::size_t found = path.find(header_names);
      path.erase(found);
    }
    else
    {
      printf("\nCooperativeGroups headers not found, please install it in %s "
             "smaple directory..\n Exiting...\n",
             argv[0]);
      exit(1);
    }
    compile_options += path.c_str();
    compile_params[num_compile_options] =
        reinterpret_cast<char *>(malloc(sizeof(char) * (compile_options.length() + 1)));

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// only support linux
#else
    snprintf(compile_params[num_compile_options],
             compile_options.size(),
             "%s",
             compile_options.c_str());
#endif

    num_compile_options++;
  }

  // compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram",
                  nvrtcCreateProgram(&prog, mem_block, filename, 0, NULL, NULL));
  nvrtcResult nvrtc_result = nvrtcCompileProgram(prog, num_compile_options, compile_params);

  // dump log
  size_t log_size;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &log_size));
  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * log_size + 1));
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
  log[log_size] = '\x0';

  if (strlen(log) >= 2)
  {
    std::cerr << "\n compilation log -- \n";
    std::cerr << log;
    std::cerr << "\n end log --\n";
  }

  free(log);

  NVRTC_SAFE_CALL("nvrtcCompileProgram", nvrtc_result);

  size_t code_size;
  NVRTC_SAFE_CALL("nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, &code_size));
  char *code =
      new char[code_size]; // same as "char *code = reinterpret_cast<char*>(malloc(sizeof(char) *
                           // code_size + 1))"
  NVRTC_SAFE_CALL("nvrtcGetCUBIN", nvrtcGetCUBIN(prog, code));
  *cubin_result = code;
  *cubin_result_size = code_size;

  for (int i = 0; i < num_compile_options; ++i)
  {
    free(compile_params[i]);
  }
}

CUmodule loadCUBIN(char *cubin, int argc, char **argv)
{
  CUmodule module;
  CUcontext context;
  int major = 0, minor = 0;
  char device_name[256];

  // pick the best CUDA device avaiable
  CUdevice cu_device = findCudaDeviceDRV(argc, (const char **) argv);

  // get compute capabilities and the device name
  checkCudaErrors(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
  checkCudaErrors(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));
  checkCudaErrors(cuDeviceGetName(device_name, 256, cu_device));
  printf("> GPU device has SM %d.%d compute capability.\n", major, minor);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuCtxCreate(&context, 0, cu_device));

  checkCudaErrors(cuModuleLoadData(&module, cubin));
  free(cubin);

  return module;
}
#endif
