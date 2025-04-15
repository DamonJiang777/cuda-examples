/*
  Multi-GPU sample using OpenMP for threading on the CPU side
  needs a compiler that supports OpenMP 2.0
*/

#include "helper_cuda.h"
#include "omp.h"
#include "stdio.h"

// a simple kernel that simply increaments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b)
{
  int idx = blockIdx.x + blockDim.x * threadIdx.x;
  g_a[idx] += b;
}

// a predicate that checks wheather each array element is set to its index plus b
int correctResult(int *data, const int n, const int b)
{
  for (int i = 0; i < n; ++i)
  {
    if (data[i] != i + b)
    {
      return 0;
    }
  }
  return 1;
}

int main(int argc, char **argv)
{
  int num_gpus = 0;

  printf("%s Starting...\n", argv[0]);

  cudaGetDeviceCount(&num_gpus);

  if (num_gpus < 1)
  {
    printf("no CUDA capable devices were detected.\n");
    exit(1);
  }

  // display cpu and gpu configuration
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for (int i = 0; i < num_gpus; ++i)
  {
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, i);
    printf("   %d: %s\n", i, dprop.name);
  }

  printf("----------------------------------\n");

  // init data
  unsigned int n = num_gpus * 8192;
  unsigned int nbytes = n * sizeof(int);
  int *a = 0;
  int b = 3;
  a = (int *) malloc(nbytes);

  if (a == 0)
  {
    printf("couldn't allocate cpu memory");
    exit(1);
  }

  for (unsigned int i = 0; i < n; ++i)
  {
    a[i] = i;
  }

  // run as many cpu threads as there are CUDA devices
  // each CPU thread controls a different device, processing its
  // portion of the data. It's possible to use more CPU threads
  // than there are CUDA devices, in which case serveral CPU
  // threads will be allocating resources and launching kernels
  // on the same device. For example, try omp_set_num_threads(2*gpus);
  // Recall that all variable declared inside an "omp parallel" scopre are
  // local to each CPU thread
  omp_set_num_threads(num_gpus); // 每个cpu 线程持有一个gpu
#pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    // set and check CUDA devices for this CPU thread
    int gpu_id = cpu_thread_id % num_gpus;
    printf("gpu_id = %d, num_gpus = %d\n", gpu_id, num_gpus);
    checkCudaErrors(cudaSetDevice(gpu_id));
    printf("CPU num threads = %d, CPU thread id = %d, CUDA device id = %d\n",
           num_cpu_threads,
           cpu_thread_id,
           gpu_id);
    checkCudaErrors(cudaGetDevice(&gpu_id));
    printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

    int *d_a = 0;
    int *sub_a = a + cpu_thread_id * n / num_cpu_threads;
    unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
    dim3 gpu_threads(128);
    dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

    checkCudaErrors(cudaMalloc((void **) &d_a, nbytes_per_kernel));
    checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
    checkCudaErrors(cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
    kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

    checkCudaErrors(cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_a));
  }
  printf("-----------------\n");

  if (cudaGetLastError() != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  }

  bool b_result = correctResult(a, n, b);

  if (a)
  {
    free(a); // free cpu memory
  }

  exit(b_result ? EXIT_SUCCESS : EXIT_FAILURE);
}
