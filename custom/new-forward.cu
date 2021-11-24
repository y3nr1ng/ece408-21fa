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
    std::cerr << "CUDA error: "
              << cudaGetErrorString(error) << ' ' << file << ':' << line << std::endl;
    if (abort) {
      exit(error);
    }
  }
}

// allocate maximal possible kernel size and reuse it between op1/2
#define M_MAX 16
#define C_MAX 4
#define KERNEL_WIDTH 7
__constant__ float kernel[M_MAX * C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];

#define TILE_WIDTH 8
#define PADDED_TILE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)
__global__ void conv_forward_kernel(float* y,
                                    const float* x,
                                    const int B,
                                    const int M,
                                    const int C,
                                    const int H,
                                    const int W,
                                    const int K) {
  extern __shared__ float tile[];

  /*
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

#define y2d(i1, i0) \
  y[b * (M * H_out * W_out) + m * (H_out * W_out) + (i1) * (W_out) + i0]
#define t3d(i2, i1, i0)                                   \
  tile[tb * (C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       (i2) * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) +   \
       (i1) * (PADDED_TILE_WIDTH) + i0]
#define x3d(i2, i1, i0) \
  x[b * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k3d(i2, i1, i0) \
  kernel[m * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Alias for block/thread index
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  // Alias for batch axis
  const int tb = threadIdx.z;
  const int b = blockIdx.z * blockDim.z + tb;

  int dst_x, dst_y, src_x, src_y;

  for (int m = 0; m < M; m++) {
    // Pre-load to shared memory, need to loop multiple time, PW^2 / W^2
    for (int c = 0; c < C; c++) {
      for (int dst = ty * TILE_WIDTH + tx;
           dst < PADDED_TILE_WIDTH * PADDED_TILE_WIDTH;
           dst += TILE_WIDTH * TILE_WIDTH) {
        // 3D index inside a padded tiles
        dst_x = dst % PADDED_TILE_WIDTH;
        dst_y = dst / PADDED_TILE_WIDTH;
        // 3D index in global array, simply subtract the pad size
        src_x = (bx * TILE_WIDTH + dst_x);
        src_y = (by * TILE_WIDTH + dst_y);

        if ((src_x < W) && (src_y < H)) {
          t3d(c, dst_y, dst_x) = x3d(c, src_y, src_x);
        } else {
          t3d(c, dst_y, dst_x) = 0.0f;
        }
      }
    }
    __syncthreads();

    // the actual convolution
    float sum = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          sum += t3d(c, ty + p, tx + q) * k3d(c, p, q);
        }
      }
    }

    // update the destination 3D index
    dst_x = bx * TILE_WIDTH + tx;
    dst_y = by * TILE_WIDTH + ty;
    // restore the linear index in global scope
    if ((dst_x < W_out) && (dst_y < H_out)) {
      y2d(dst_y, dst_x) = sum;
    }
    __syncthreads();
  }

#undef y2d
#undef t3d
#undef x3d
#undef k3d
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
  std::cout << "*** constant memory + tiled shared memory ***" << std::endl;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);

  printf("(B=%d, M=%d, C=%d, H=%d, W=%d, K=%d)\n", B, M, C, H, W, K);

  // Allocate memory
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));

  // Copy over the relevant data structures to the GPU
  cudaErrChk(cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpyToSymbol(kernel, host_k, bytes_k));
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
  // Set the kernel dimensions
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Batch across batch dimension
  const int B_batch = 8;

  // Calculate launch size
  dim3 block(TILE_WIDTH, TILE_WIDTH, B_batch);
  dim3 grid(ceil((float)W_out / TILE_WIDTH),
            ceil((float)H_out / TILE_WIDTH),
            ceil((float)B / B_batch));
  printf("grid=(x=%d, y=%d, z=%d), block=(x=%d, y=%d, z=%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  // Determine shared memory size
  size_t smem_size =
      B_batch * C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH * sizeof(float);

  // Call the kernel
  conv_forward_kernel<<<grid, block, smem_size>>>(device_y, device_x, B, M, C, H, W, K);
  cudaErrChk(cudaDeviceSynchronize());
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
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);

  // Copy the output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_x));
}

__host__ void GPUInterface::get_device_properties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational capabilities: "
              << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem
              << std::endl;
    std::cout << "Max Constant memory size: " << deviceProp.totalConstMem
              << std::endl;
    std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock
              << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max block dimensions: "
              << deviceProp.maxThreadsDim[0] << " x, "
              << deviceProp.maxThreadsDim[1] << " y, "
              << deviceProp.maxThreadsDim[2] << " z" << std::endl;
    std::cout << "Max grid dimensions: "
              << deviceProp.maxGridSize[0] << " x, "
              << deviceProp.maxGridSize[1] << " y, "
              << deviceProp.maxGridSize[2] << " z" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
  }
}
