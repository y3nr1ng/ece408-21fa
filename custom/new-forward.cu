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
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << ' ' << file
              << ':' << line << std::endl;
    if (abort) {
      exit(error);
    }
  }
}

#define M_MAX 16
#define C_MAX 4
#define KERNEL_WIDTH 7
__constant__ float kernel[M_MAX * C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];

#define KERNEL_RADIUS (KERNEL_WIDTH / 2)
#define TILE_WIDTH 8
#define PADDED_TILE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)
__global__ void conv_forward_kernel(float* y,
                                    const float* x,
                                    const int b,
                                    const int m,
                                    const int M,
                                    const int C,
                                    const int H,
                                    const int W,
                                    const int K) {
  __shared__ float
      tile[PADDED_TILE_WIDTH * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH];

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

#define y3d(i1, i0) \
  y[b * (M * H_out * W_out) + m * (H_out * W_out) + (i1) * (W_out) + i0]
#define t3d(i2, i1, i0)                                 \
  tile[(i2) * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       (i1) * (PADDED_TILE_WIDTH) + i0]
#define x3d(i2, i1, i0) x[b * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k3d(i2, i1, i0) \
  kernel[m * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  int tmp;

  // some alias for block/thread index
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  // destination linear index
  int dst = (tz * TILE_WIDTH * TILE_WIDTH) + (ty * TILE_WIDTH) + tx;
  // 3D index inside a padded tiles
  tmp = dst;
  int dst_x = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  int dst_y = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  int dst_z = tmp;
  // 3D index in global array, simply subtract the pad size
  int src_x = (bx * TILE_WIDTH + dst_x) - KERNEL_RADIUS;
  int src_y = (by * TILE_WIDTH + dst_y) - KERNEL_RADIUS;
  int src_z = (bz * TILE_WIDTH + dst_z) - KERNEL_RADIUS;

  // load 1, this include "left halos" and "content"
  if (((src_x >= 0) && (src_x < W)) && ((src_y >= 0) && (src_y < H)) &&
      ((src_z >= 0) && (src_z < C))) {
    t3d(dst_z, dst_y, dst_x) = x3d(src_z, src_y, src_x);
  } else {
    t3d(dst_z, dst_y, dst_x) = 0.0f;
  }

  // load 2, "right halos",
  dst = (tz * TILE_WIDTH * TILE_WIDTH) + (ty * TILE_WIDTH) + tx +
        (TILE_WIDTH * TILE_WIDTH * TILE_WIDTH);
  tmp = dst;
  dst_x = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  dst_y = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  dst_z = tmp;
  src_x = (bx * TILE_WIDTH + dst_x) - KERNEL_RADIUS;
  src_y = (by * TILE_WIDTH + dst_y) - KERNEL_RADIUS;
  src_z = (bz * TILE_WIDTH + dst_z) - KERNEL_RADIUS;
  if (dst_z < PADDED_TILE_WIDTH) {
    if (((src_x >= 0) && (src_x < W)) &&
        ((src_y >= 0) && (src_y < H)) && ((src_z >= 0) && (src_z < C))) {
      t3d(dst_z, dst_y, dst_x) = x3d(src_z, src_y, src_x);
    } else {
      t3d(dst_z, dst_y, dst_x) = 0.0f;
    }
  }

  __syncthreads();

  // update the destination 3D index
  dst_x = bx * TILE_WIDTH + tx;
  dst_y = by * TILE_WIDTH + ty;
  dst_z = bz * TILE_WIDTH + tz;

  if (dst_z == 0) {
    // the actual convolution
    float sum = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          sum += t3d(tz + c, ty + p, tx + q) * k3d(c, p, q);
        }
      }
    }

    // restore the linear index in global scope
    if ((dst_x < W_out) && (dst_y < H_out)) {
      y3d(dst_y, dst_x) = sum;
    }
  }

#undef y3d
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

  // Allocate memory
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));

  // Copy over the relevant data structures to the GPU
  cudaErrChk(
      cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
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

  dim3 block(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  dim3 grid(ceil((float)H_out / TILE_WIDTH), ceil((float)W_out / TILE_WIDTH),
            ceil((float)C / TILE_WIDTH));

  std::cout << "grid=(" << grid.x << ", " << grid.y << ", " << grid.z << "), "
            << "block=(" << block.x << ", " << block.y << ", " << block.z
            << ") " << std::endl;

  // Call the kernel
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      conv_forward_kernel<<<grid, block>>>(device_y, device_x, b, m, M, C, H, W,
                                           K);
      cudaErrChk(cudaDeviceSynchronize());
    }
  }
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
