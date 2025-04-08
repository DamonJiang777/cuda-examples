/*
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel accurately. Blocks are executed in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "helper_cuda.h"

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in the device memory
__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0)
  {
    timer[bid] = clock();
  }

  // copy input
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  // perform reduction to find minmum
  for (int d = blockDim.x; d > 0; d /= 2)
  {
    __syncthreads(); // sync write input

    if (tid < d) // can't remove
    {
      float f0 = shared[tid];
      float f1 = shared[tid + d];

      if (f1 < f0)
      {
        shared[tid] = f1;
      }
    }
  }

  // write result
  if (tid == 0)
  {
    output[bid] = shared[0];
  }

  __syncthreads();

  if (tid == 0)
  {
    timer[bid + gridDim.x] = clock();
  }
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char **argv)
{
  printf("CUDA clock sample.\n");

  int dev = FindCudaDevice(argc, (const char **) argv);

  float *d_input = NULL;
  float *d_output = NULL;
  clock_t *d_timer = NULL;

  clock_t timer[NUM_BLOCKS * 2];
  float input[NUM_THREADS * 2];

  for (int i = 0; i < NUM_THREADS * 2; ++i)
  {
    input[i] = (float) i;
  }

  checkCudaErrors(cudaMalloc((void **) &d_input, sizeof(float) * NUM_THREADS * 2));
  checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float) * NUM_BLOCKS));
  checkCudaErrors(cudaMalloc((void **) &d_timer, sizeof(clock_t) * NUM_BLOCKS * 2));

  checkCudaErrors(
      cudaMemcpy(d_input, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));
  timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(d_input,
                                                                               d_output,
                                                                               d_timer);

  checkCudaErrors(
      cudaMemcpy(timer, d_timer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_timer));

  long double avg_elapsed_clocks = 0;
  for (int i = 0; i < NUM_BLOCKS; ++i)
  {
    avg_elapsed_clocks += (long double) (timer[i + NUM_BLOCKS] - timer[i]);
  }

  avg_elapsed_clocks = avg_elapsed_clocks / NUM_BLOCKS;
  printf("Average clocks/block = %Lf\n", avg_elapsed_clocks);

  return EXIT_SUCCESS;
}
