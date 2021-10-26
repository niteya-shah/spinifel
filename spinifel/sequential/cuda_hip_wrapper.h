#ifndef CUDA_HIP_WRAPPER_H
#define CUDA_HIP_WRAPPER_H

#ifdef USE_HIP

#include <hip/hip_runtime.h>

#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaSetDevice hipSetDevice
#define cudaFree hipFree
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#endif

#endif
