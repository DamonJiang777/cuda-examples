#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include "helper_cuda.h"
#include "helper_timer.h"

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] += inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
  for (int i = 0; i < n; ++i)
  {
    if (data[i] != x)
    {
      printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv)
{
  int            device_id;
  cudaDeviceProp device_props;

  printf("[%s] - Starting...\n", argv[0]);

  device_id = FindCudaDevice(argc, (const char **) argv);

  checkCudaErrors(cudaGetDeviceProperties(&device_props, device_id));
  printf("CUDA device [%s]\n", device_props.name);

  int n      = 16 * 1024 * 1024;
  int nbytes = n * sizeof(int);
  int value  = 26;

  // allocate host memory
  int *a = 0;
  checkCudaErrors(cudaMallocHost((void **) &a, nbytes));
  memset(a, 0, nbytes);

  // allocate device memory
  int *d_a = 0;
  checkCudaErrors(cudaMalloc((void **) &d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(n / threads.x, 1);

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU(all to stream 0)
  checkCudaErrors(cudaProfilerStart());
  sdkStartTimer(&timer);
  cudaEventRecord(start, 0);                                  // stream 0
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0); // stream 0
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);    // add value
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);
  checkCudaErrors(cudaProfilerStop());

  // have CPU do some work while waiting for stage 1 to finish
  unsigned long int counter = 0;
  while (cudaEventQuery(stop) == cudaErrorNotReady)
  {
    counter++;
  }

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

  // print the cpu and gpu times
  printf("time spent executing by the GPU: %.2f\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
  printf("CPU executed %lu iterations while waitting for the GPU to finish\n", counter);

  // check the output for correctness
  bool b_final_results = correct_output(a, n, value);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));

  exit(b_final_results ? EXIT_SUCCESS : EXIT_FAILURE);
  return 0;
}
