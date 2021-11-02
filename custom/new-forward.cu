#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define cudaErrChk(stmt) \
  { cudaAssert((stmt), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t error,
                       const char* file,
                       int line,
                       bool abort = true) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << file << line
              << std::endl;
    if (abort) {
      exit(error);
    }
  }
}

__global__ void conv_forward_kernel(float* y,
                                    const float* x,
                                    const float* k,
                                    const int B,
                                    const int M,
                                    const int C,
                                    const int H,
                                    const int W,
                                    const int K) {
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire
  mini-batch The goal here is to be correct AND fast.

  Function paramter definitions:
  y - output
  x - input
  k - kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  (void)H_out;  // silence declared but never referenced warning. remove this
                // line when you start working
  (void)W_out;  // silence declared but never referenced warning. remove this
                // line when you start working

  // We have some nice #defs for you below to simplify indexing. Feel free to
  // use them, or create your own. An example use of these macros: float a =
  // y4d(0,0,0,0) y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) \
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) \
  x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) \
  k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your GPU convolution kernel code here

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float* host_y,
                                                    const float* host_x,
                                                    const float* host_k,
                                                    float** device_y_ptr,
                                                    float** device_x_ptr,
                                                    float** device_k_ptr,
                                                    const int B,
                                                    const int M,
                                                    const int C,
                                                    const int H,
                                                    const int W,
                                                    const int K) {
  /*
  y - output
  x - input
  k - kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (C * H * W) * sizeof(float);
  const size_t bytes_k = (C * K * K) * sizeof(float);

  // Allocate memory
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));
  cudaErrChk(cudaMalloc(device_k_ptr, bytes_k));

  // Copy over the relevant data structures to the GPU
  cudaErrChk(cudaMemcpy(device_y_ptr, host_y, bytes_y, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(device_k_ptr, host_k, bytes_k, cudaMemcpyHostToDevice));
}

__host__ void GPUInterface::conv_forward_gpu(float* device_y,
                                             const float* device_x,
                                             const float* device_k,
                                             const int B,
                                             const int M,
                                             const int C,
                                             const int H,
                                             const int W,
                                             const int K) {
  // Set the kernel dimensions and call the kernel
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_y,
                                                    float* device_y,
                                                    float* device_x,
                                                    float* device_k,
                                                    const int B,
                                                    const int M,
                                                    const int C,
                                                    const int H,
                                                    const int W,
                                                    const int K) {
  /*
  y - output
  x - input
  k - kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (M * H_out * W_out) * sizeof(float);

  // Copy the output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_x));
  cudaErrChk(cudaFree(device_k));
}

__host__ void GPUInterface::get_device_properties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational capabilities: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem
              << std::endl;
    std::cout << "Max Constant memory size: " << deviceProp.totalConstMem
              << std::endl;
    std::cout << "Max Shared memory size per block: "
              << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0]
              << " x, " << deviceProp.maxThreadsDim[1] << " y, "
              << deviceProp.maxThreadsDim[2] << " z" << std::endl;
    std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
              << deviceProp.maxGridSize[1] << " y, "
              << deviceProp.maxGridSize[2] << " z" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
  }
}
