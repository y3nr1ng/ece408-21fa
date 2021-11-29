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

// Some feature flags
//#define USE_STREAM // Use multi-stream to accelerate transfers
//#define USE_ASYNC_ALLOCATOR // Use async allocators, available >= 11.2

// Allocate maximal possible kernel size and reuse it between op1/2
#define M_MAX 16
#define C_MAX 4
#define KERNEL_WIDTH 7
__constant__ float kernel[M_MAX * C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];

// Tile configurations
#define TILE_WIDTH 8
#define PADDED_TILE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)

/*
  KERNELS_NUM = L * M * C;

  if (gridSize == 0)
    gridSize = ceil((float)KERNELS_NUM / blockSize);

  im2colOnDevice<<<gridSize, blockSize>>>(
    KERNELS_NUM, devAc, devA, radiusF, countF, L, M, K, C);

  input, A.shape = C, H, W
  kernel, F.shape = (R=C, Q=K, P=K), D=1 (# kernels)
  output, B.shape = N=D=1, M=W*(K-1), L=H*(K-1)

  countF = K*K*C
  radiusF = (K-1)/2
*/

// Prepare input features as column matrix
__global__ void im2col(float* xc,
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

  // Y = (H W) * (K^2 C)
  /*
      c - input feature map
  ho/wo - output height/width
  hi/wi - input height/width
  hk/wk - convolution loop height/width
  */
#define xc5d(i_ho, i_wo, i_c, i_hk, i_wk)        \
  xc[(b) * (H_out * W_out * C * K * K) +         \
     ((i_ho) * (W_out) + (i_wo)) * (C * K * K) + \
     (i_c) * (K * K) + (i_hk) * (K) + (i_wk)]
#define t3d(i_c, i_ph, i_pw)                                \
  tile[(tb) * (C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       (i_c) * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) +    \
       (i_ph) * (PADDED_TILE_WIDTH) +                       \
       (i_pw)]
#define x3d(i_c, i_hi, i_wi) \
  x[(b) * (C * H * W) +      \
    (i_c) * (H * W) +        \
    (i_hi) * (W) +           \
    (i_wi)]

  // Alias for block/thread index
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  // Alias for batch axis
  const int tb = threadIdx.z;
  const int b = blockIdx.z * blockDim.z + tb;

  int dst_x, dst_y, src_x, src_y;

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

  // Update destination location
  dst_x = bx * TILE_WIDTH + tx;
  dst_y = by * TILE_WIDTH + ty;

  // Flatten out the matrix for current output pixel
  if ((dst_x < W_out) && (dst_y < H_out)) {
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < KERNEL_WIDTH; p++) {
        for (int q = 0; q < KERNEL_WIDTH; q++) {
          xc5d(dst_y, dst_x, c, p, q) = t3d(c, ty + p, tx + q);
        }
      }
    }
  }

#undef xc5d
#undef t3d
#undef x3d
}

__global__ void matrix_multiply(float* y,
                                const float* xc,
                                const int B,
                                const int M,
                                const int C,
                                const int H,
                                const int W,
                                const int K) {
  extern __shared__ float tile[];

  /*
   y.shape = (B, M, H_out, W_out)
  xc.shape = (H_out * W_out, C * K * K) = (H_out, W_out, C * K * K)
   t.shape = (b, 0/1, H_out, W_out, C, K, K)
   k.shape = (M, C * K * K)
  */
#define y2d(i_m, i_hw)          \
  y[(b) * (M * H_out * W_out) + \
    (i_m) * (H_out * W_out) +   \
    (i_hw)]
#define xc2d(i_hw, i_ckk)                \
  xc[(b) * (H_out * W_out * C * K * K) + \
     (i_hw) * (C * K * K) +              \
     (i_ckk)]
#define t2d(i, i_hw, i_ckk)                   \
  tile[(tb) * (2 * TILE_WIDTH * TILE_WIDTH) + \
       (i) * (TILE_WIDTH * TILE_WIDTH) +      \
       (i_hw) * (TILE_WIDTH) +                \
       (i_ckk)]
#define k2d(i_m, i_ckk)        \
  kernel[(i_m) * (C * K * K) + \
         (i_ckk)]

#define t2d_xc(i_hw, i_ckk) \
  t2d(0, i_hw, i_ckk)
#define t2d_kt(i_m, i_ckk) \
  t2d(1, i_m, i_ckk)

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int n_hw = H_out * W_out;

  // Alias for block/thread index
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  // Alias for batch axis
  const int tb = threadIdx.z;
  const int b = blockIdx.z * blockDim.z + tb;

  // Identify the row/column of output element
  // TODO currently, share TILE_WIDTH with im2col, Tensor needs TILE_WIDTH=16
  int dst_hw = bx * TILE_WIDTH + tx;  // col
  int dst_m = by * TILE_WIDTH + ty;   // row

  // Calculate number of subtiles
  const int n_kernel = C * K * K;
  const int n_tiles = (n_kernel + (TILE_WIDTH - 1)) / TILE_WIDTH;

  int dst_ckk;

  float acc = 0;
  for (int n = 0; n < n_tiles; n++) {
    // Save sub-tile of xc and kernel to smem
    dst_ckk = n * TILE_WIDTH + ty;
    if ((dst_hw < n_hw) && (dst_ckk < n_kernel)) {
      t2d_xc(ty, tx) = xc2d(dst_hw, dst_ckk);
    } else {
      t2d_xc(ty, tx) = 0.0;
    }
    dst_ckk = n * TILE_WIDTH + tx;
    if ((dst_m < M) && (dst_ckk < n_kernel)) {
      t2d_kt(ty, tx) = k2d(dst_m, dst_ckk);
    } else {
      t2d_kt(ty, tx) = 0.0;
    }
    __syncthreads();

    // C_ij = A_ik * B_kj ===> C_ij^T = B_kj^T * A_ik^T
    for (int k = 0; k < TILE_WIDTH; k++) {
      acc += t2d_xc(k, tx) * t2d_kt(ty, k);
    }
    __syncthreads();
  }

  if ((dst_m < M) && (dst_hw < n_hw)) {
    y2d(dst_m, dst_hw) = acc;
  }

#undef t2d_xc
#undef t2d_kt

#undef y2d
#undef xc2d
#undef t2d
#undef k2d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float* host_y,
                                                    const float* host_x,
                                                    const float* host_k,
                                                    float** device_y_ptr,
                                                    float** device_xc_ptr,
                                                    float** device_k_ptr,
                                                    const int B,
                                                    const int M,
                                                    const int C,
                                                    const int H,
                                                    const int W,
                                                    const int K) {
  std::cout << "*** constant mem + tiled + restrict/unroll + gemm ***" << std::endl;
  printf("(B=%d, M=%d, C=%d, H=%d, W=%d, K=%d)\n", B, M, C, H, W, K);

  // Buffer for im2col
  float* device_x;

  // Estimat output dimension
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  printf("(H_out=%d, W_out=%d)\n", H_out, W_out);

  // Block size along the B (batch) dimension
  const int B_batch_size = 4;

  // Calculate needed bytes for original input
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);
  // Calculate needed bytes for unrolled input
  const size_t bytes_xc = (B * (H_out * W_out) * (C * K * K)) * sizeof(float);

  const float ratio = (float)bytes_xc / bytes_x;
  const float bytes_xc_mb = (float)bytes_xc / 1024 / 1024;
  printf("*** memory increased %.2fx, %.2fMiB\n", ratio, bytes_xc_mb);

  // Allocate memory on device
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(&device_x, bytes_x));
  cudaErrChk(cudaMalloc(device_xc_ptr, bytes_xc));

  // Copy input data to device
  cudaErrChk(cudaMemcpy(device_x, host_x, bytes_x, cudaMemcpyHostToDevice));

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

  // Unroll to column matrix
  im2col<<<grid, block, smem_size>>>(*device_xc_ptr,
                                     device_x,
                                     B, M, C, H, W, K);

  // Copy kernel weights
  cudaErrChk(cudaMemcpyToSymbol(kernel, host_k, bytes_k));

  // Free input buffer
  cudaErrChk(cudaFree(device_x));
}

__host__ void GPUInterface::conv_forward_gpu(float* device_y,
                                             const float* device_xc,
                                             const float* device_k,  // unused
                                             const int B,
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

  // Calculate launch size
  dim3 block(TILE_WIDTH, TILE_WIDTH, B_batch_size);
  dim3 grid(ceil((float)H_out * W_out / block.x),
            ceil((float)M / block.y),
            ceil((float)B / block.z));
  printf("grid=(x=%d, y=%d, z=%d), block=(x=%d, y=%d, z=%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  // Determine shared memory size
  size_t smem_size =
      B_batch_size * 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

  // Call the kernel
  matrix_multiply<<<grid, block, smem_size>>>(device_y, device_xc, B, M, C, H, W, K);
  cudaErrChk(cudaDeviceSynchronize());
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_y,
                                                    float* device_y,
                                                    float* device_xc,
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

  // Copy output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_xc));
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
