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

// Some flags
#define USE_STREAM // Use multi-stream to accelerate transfers

// Allocate maximal possible kernel size and reuse it between op1/2
#define M_MAX 16
#define C_MAX 4
#define KERNEL_WIDTH 7
__constant__ float kernel[M_MAX * C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];

// The actual convolution kernel
#define TILE_WIDTH 8
#define PADDED_TILE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)
__global__ void conv_forward_kernel(float* __restrict__ y,
                                    const float* __restrict__ x,
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
#pragma unroll
      for (int p = 0; p < KERNEL_WIDTH; p++) {
#pragma unroll
        for (int q = 0; q < KERNEL_WIDTH; q++) {
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
  std::cout << "*** constant mem + tiled + restrict/unroll + stream ***" << std::endl;

  printf("(B=%d, M=%d, C=%d, H=%d, W=%d, K=%d)\n", B, M, C, H, W, K);

  // Estimat output dimension
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Calculate needed bytes
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);

#ifndef USE_STREAM
  // Allocate memory on device
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));

  // Copy input data to device
  cudaErrChk(cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
#else
  // Pass through host pointers
  *device_y_ptr = (float*)host_y;
  *device_x_ptr = (float*)host_x;

  // Mark them as pinned memory for asynchronous transfer
  cudaErrChk(cudaHostRegister(*device_y_ptr, bytes_y, cudaHostRegisterPortable));
  cudaErrChk(cudaHostRegister(*device_x_ptr, bytes_x, cudaHostRegisterPortable));
#endif

  // Copy kernel weights
  cudaErrChk(cudaMemcpyToSymbol(kernel, host_k, bytes_k));
}

__host__ void GPUInterface::conv_forward_gpu(float* device_y,
                                             const float* device_x,
                                             const float* device_k,  // unused
                                             const int B0,
                                             const int M,
                                             const int C,
                                             const int H,
                                             const int W,
                                             const int K) {
  // Estimat output dimension
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Block size along the B (batch) dimension
  const int B_batch_size = 4;

  /*** Prolog BEGIN ***/
#ifndef USE_STREAM
  // Send the entire batch
  const int B = B0;
#else
  // Create streams
  const int n_streams = 16;
  cudaStream_t stream[n_streams];
  for (int i = 0; i < n_streams; i++) {
    cudaErrChk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
  }

  // We pass through host pointers from the prolog function
  const float* host_x = device_x;
  float* host_y = device_y;

  // Calculate total elements and bytes
  const int n_y = B0 * M * H_out * W_out;
  const int n_x = B0 * C * H * W;
  const size_t bytes_y = n_y * sizeof(float);
  const size_t bytes_x = n_x * sizeof(float);
  // Calculate partial elements and bytes per stream
  const int B = ceil((float)B0 / n_streams);
  const int n_y_stream = B * M * H_out * W_out;
  const int n_x_stream = B * C * H * W;

  // Allocate memory
  // TODO use cudaMallocAsync
  cudaErrChk(cudaMalloc(&device_y, bytes_y));
  cudaErrChk(cudaMalloc(&device_x, bytes_x));

  // Copy over the relevant data structures to the GPU
  for (int i = 0; i < n_streams; i++) {
    size_t offset = i * n_x_stream;
    size_t bytes = n_x_stream * sizeof(float);
    if (offset + n_x_stream > n_x) {
      // Last stream does not need to copy that much
      bytes = (n_x - offset) * sizeof(float);
    }
    cudaErrChk(cudaMemcpyAsync((void*)&device_x[offset], &host_x[offset], bytes,
                               cudaMemcpyHostToDevice, stream[i]));
  }
#endif
  /*** Prolog END ***/


  /*** Kernel call BEGIN ***/
  // Calculate launch size
  dim3 block(TILE_WIDTH, TILE_WIDTH, B_batch_size);
  dim3 grid(ceil((float)W_out / block.x),
            ceil((float)H_out / block.y),
            ceil((float)B / block.z));
  printf("grid=(x=%d, y=%d, z=%d), block=(x=%d, y=%d, z=%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  // Determine shared memory size
  size_t smem_size =
      B_batch_size * C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH * sizeof(float);

  // Call the kernel
#ifndef USE_STREAM
  conv_forward_kernel<<<grid, block, smem_size>>>(device_y, device_x, B, M, C, H, W, K);
  cudaErrChk(cudaDeviceSynchronize());
#else
  for (int i = 0; i < n_streams; i++) {
    size_t offset_y = i * n_y_stream;
    size_t offset_x = i * n_x_stream;
    conv_forward_kernel<<<grid, block, smem_size, stream[i]>>>(
        &device_y[offset_y], &device_x[offset_x], B, M, C, H, W, K);
  }
#endif
  /*** Kernel call END ***/


  /*** Epilog BEGIN ***/
#ifndef USE_STREAM
  // We directly wait for the single kernel to end
  cudaErrChk(cudaDeviceSynchronize());
#else
  // Copy back data to host
  for (int i = 0; i < n_streams; i++) {
    size_t offset = i * n_y_stream;
    size_t bytes = n_y_stream * sizeof(float);
    if (offset + n_y_stream > n_y) {
      // Last stream does not need to copy that much
      bytes = (n_y - offset) * sizeof(float);
    }
    cudaErrChk(cudaMemcpyAsync(&host_y[offset], &device_y[offset], bytes,
                               cudaMemcpyDeviceToHost, stream[i]));
  }

  // Need to wait every stream to finish before freeing up memory
  cudaErrChk(cudaDeviceSynchronize());

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree((void*)device_x));

  // Destory streams
  for (int i = 0; i < n_streams; i++) {
    cudaErrChk(cudaStreamDestroy(stream[i]));
  }
#endif
  /*** Epilog END ***/
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
#ifndef USE_STREAM
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);

  // Copy output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_x));
#else
  // Data is already write back to host earlier, safe to clean up now

  // Release pinned memory
  cudaErrChk(cudaHostUnregister(device_y));
  cudaErrChk(cudaHostUnregister(device_x));
#endif
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
