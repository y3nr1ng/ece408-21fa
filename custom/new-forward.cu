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
#define xc5d(i_ho, i_wo, i_c, i_hk, i_wk)    \
  xc[b * (H_out * W_out * C * k * K) +       \
     (i_ho * (W_out) + i_wo) * (C * K * K) + \
     i_c * (K * K) + i_hk * (K) + i_wk]
#define t3d(i_c, i_ph, i_pw)                              \
  tile[tb * (C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       i_c * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) +    \
       i_ph * (PADDED_TILE_WIDTH) +                       \
       i_pw]
#define x3d(i_c, i_hi, i_wi) \
  x[b * (C * H * W) +        \
    i_c * (H * W) +          \
    i_hi * (W) +             \
    i_wi]

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

// TODO get back to work from here
__global__ void matrix_multiply(float* y,
                                const float* xc,
                                const int B,
                                const int M,
                                const int C,
                                const int H,
                                const int W,
                                const int K) {
  extern __shared__ float sub_tiles[];

#define y2d(i1, i0) \
  y[b * (M * H_out * W_out) + m * (H_out * W_out) + (i1) * (W_out) + i0]
#define xc5d(i_ho, i_wo, i_c, i_hk, i_wk)    \
  xc[b * (H_out * W_out * C * k * K) +       \
     (i_ho * (W_out) + i_wo) * (C * K * K) + \
     i_c * (K * K) + i_hk * (K) + i_wk]

  // original im2col implementation
#define xc5d(i_ho, i_wo, i_c, i_hk, i_wk)    \
  xc[b * (H_out * W_out * C * k * K) +       \
     (i_ho * (W_out) + i_wo) * (C * K * K) + \
     i_c * (K * K) + i_hk * (K) + i_wk]
#define t3d(i_c, i_ph, i_pw)                              \
  tile[tb * (C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       i_c * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) +    \
       i_ph * (PADDED_TILE_WIDTH) +                       \
       i_pw]
#define x3d(i_c, i_hi, i_wi) \
  x[b * (C * H * W) +        \
    i_c * (H * W) +          \
    i_hi * (W) +             \
    i_wi]

// original stream implementation
#define t3d(i2, i1, i0)                                   \
  tile[tb * (C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) + \
       (i2) * (PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) +   \
       (i1) * (PADDED_TILE_WIDTH) + i0]
#define x3d(i2, i1, i0) \
  x[b * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k3d(i2, i1, i0) \
  kernel[m * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
}

// Compute C = A * B
__global__ void matrixMultiplyShared(
    float* A,
    float* B,
    float* C,
    int numARows,
    int numAColumns,
    int numBRows,
    int numBColumns,
    int numCRows,
    int numCColumns) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  // abbreviations for thread index
  int tx = threadIdx.x, ty = threadIdx.y;

  // identify the row and column of the C element to work on
  int column = blockIdx.x * TILE_WIDTH + tx;
  int row = blockIdx.y * TILE_WIDTH + ty;

  // loop over A and B tiles required to compute C
  // A_ik * B_kj = C_ij
  float sum = 0;
  int nTiles = (numAColumns + (TILE_WIDTH - 1)) / TILE_WIDTH;
  for (int n = 0; n < nTiles; n++) {
    if (n * TILE_WIDTH + tx < numAColumns) {
      subTileA[ty][tx] = A[row * numAColumns + (n * TILE_WIDTH + tx)];
    } else {
      subTileA[ty][tx] = 0.0;
    }
    if (n * TILE_WIDTH + ty < numBRows) {
      subTileB[ty][tx] = B[(n * TILE_WIDTH + ty) * numBColumns + column];
    } else {
      subTileB[ty][tx] = 0.0;
    }
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      sum += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }
  if ((column < numCColumns) && (row < numCRows)) {
    C[row * numCColumns + column] = sum;
  }
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

  // Block size along the B (batch) dimension
  const int B_batch_size = 4;

  // Calculate needed bytes for original input
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);
  // Calculate needed bytes for unrolled input
  const size_t bytes_xc = (B * (H_out * W_out) * (C * K * K)) * sizeof(float);

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
                                             const float* device_x,
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
  dim3 grid(ceil((float)W_out / block.x),
            ceil((float)H_out / block.y),
            ceil((float)B / block.z));
  printf("grid=(x=%d, y=%d, z=%d), block=(x=%d, y=%d, z=%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  // Determine shared memory size
  size_t smem_size =
      B_batch_size * C * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH * sizeof(float);

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

  // Copy output back to host
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
