#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// ElementWise Add grid(N/256)
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c:Nx1, c = elementwise_add_f32x4_kernel(a, b)
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}

// FP16
// Elementwise Add gird(N/256)
// block(256) a: Nx1, b: Nx1, 
