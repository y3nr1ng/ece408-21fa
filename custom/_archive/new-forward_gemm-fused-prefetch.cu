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

// Stream configurations
//#define USE_STREAM  // Use multi-stream to accelerate transfers
#define N_STREAMS 16

// Allocate maximal possible kernel size and reuse it between op1/2
#define M_MAX 16
#define C_MAX 4
#define KERNEL_WIDTH 7
__constant__ float kernel[M_MAX * C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];
__constant__ int3 conv_lut[C_MAX * KERNEL_WIDTH * KERNEL_WIDTH];

// Tile configurations
#define TILE_WIDTH 8
#define PADDED_TILE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)

// NOTE:
//  B_BATCH * [
//    (HW_TILE_WIDTH + M_TILE_WIDTH) * CKK_TILE_WIDTH
//    + TILE_WIDTH * TILE_WIDTH
//  ]
#define HW_TILE_WIDTH (TILE_WIDTH * TILE_WIDTH)
#define M_TILE_WIDTH 4
#define CKK_TILE_WIDTH 16

// Block size along the B (batch) dimension
#define B_BATCH 4

__global__ void conv_as_gemm(float* __restrict__ y,
                             const float* __restrict__ x,
                             const int B,
                             const int M,
                             const int C,
                             const int H,
                             const int W,
                             const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Alias for height/width axis
  const int t_hw = threadIdx.x, b_hw = HW_TILE_WIDTH;
  // Alias for output channels
  const int t_m = threadIdx.y, b_m = M_TILE_WIDTH;
  const int m = blockIdx.y * b_m + t_m;
  // Alias for batch axis
  const int t_b = threadIdx.z;
  const int b = blockIdx.z * blockDim.z + t_b;

  // Y = (H W) * (K^2 C)
  /*
      c - input feature map
  ho/wo - output height/width
  hi/wi - input height/width
  hk/wk - convolution loop height/width
  */

  /*
   y.shape = (B, M, H_out, W_out)
   t.shape = (b, 0/1, H_out, W_out, C, K, K)
   k.shape = (M, C * K * K)
  */
#define y1d(i_hw)               \
  y[(b) * (M * H_out * W_out) + \
    (m) * (H_out * W_out) +     \
    (i_hw)]
#define x3d(i_c, i_ih, i_iw) \
  x[(b) * (C * H * W) +      \
    (i_c) * (H * W) +        \
    (i_ih) * (W) +           \
    (i_iw)]
#define k1d(i_ckk)           \
  kernel[(m) * (C * K * K) + \
         (i_ckk)]

  __shared__ float tile_xc[B_BATCH][HW_TILE_WIDTH][CKK_TILE_WIDTH + 1];
  __shared__ float tile_kt[B_BATCH][M_TILE_WIDTH][CKK_TILE_WIDTH + 1];
  __shared__ float tile_x[B_BATCH][C_MAX][PADDED_TILE_WIDTH][PADDED_TILE_WIDTH + 1];

#define t2d_kt(i_m, i_ckk) \
  tile_kt[t_b][i_m][i_ckk]
#define t2d_xc(i_ckk, i_hw) \
  tile_xc[t_b][i_hw][i_ckk]

  const int n_hw = H_out * W_out;

  // Aggregate all threads, we will reassign their index
  // NOTE: HW_TILE_WIDTH * M_TILE_WIDTH = TILE_WIDTH * TILE_WIDTH

  const int ib_hw = blockIdx.x;
  const int nb_w = (W_out + (TILE_WIDTH - 1)) / TILE_WIDTH;
  // .. Block index in 2D grid
  const int ib_w = ib_hw % nb_w;
  const int ib_h = ib_hw / nb_w;
  // .. Thread index in 2D block
  const int t_w = t_hw % TILE_WIDTH;
  const int t_h = t_hw / TILE_WIDTH;

  const int dst_w = ib_w * TILE_WIDTH + t_w;
  const int dst_h = ib_h * TILE_WIDTH + t_h;
  const int dst_hw = dst_h * W_out + dst_w;

  // Calculate number of subtiles
  const int n_kernel = C * K * K;
  const int n_tiles = (n_kernel + (CKK_TILE_WIDTH - 1)) / CKK_TILE_WIDTH;

  int t_pw, t_ph, src_w, src_h;
  if ((b < B)) {
    // Pre-load to shared memory, need to loop multiple time, PW^2 / W^2
    for (int c = 0; c < C; c++) {
      for (int dst = t_m * HW_TILE_WIDTH + t_hw;
           dst < PADDED_TILE_WIDTH * PADDED_TILE_WIDTH;
           dst += TILE_WIDTH * TILE_WIDTH) {
        // 2D block index in a padded tile
        t_pw = dst % PADDED_TILE_WIDTH;
        t_ph = dst / PADDED_TILE_WIDTH;
        // 3D index in global array, simply subtract the pad size
        src_w = ib_w * TILE_WIDTH + t_pw;
        src_h = ib_h * TILE_WIDTH + t_ph;

        if ((src_w < W) && (src_h < H)) {
          tile_x[t_b][c][t_ph][t_pw] = x3d(c, src_h, src_w);
        } else {
          tile_x[t_b][c][t_ph][t_pw] = 0.0f;
        }
      }
    }
    __syncthreads();

    float acc = 0;
    for (int n = 0; n < n_tiles; n++) {
      // Save sub-tile of xc and kernel to smem
      for (int i_ckk = t_m; i_ckk < CKK_TILE_WIDTH; i_ckk += M_TILE_WIDTH) {
        int dst_ckk = n * CKK_TILE_WIDTH + i_ckk;
        if ((dst_w < W_out) && (dst_h < H_out) && (dst_ckk < n_kernel)) {
          // Do input matrix unroll on-the-fly
          int3 lut = conv_lut[dst_ckk];
          const int q = lut.x, p = lut.y, c = lut.z;

          // Load from global memory to tiled memory as unrolled column matrix
          tile_xc[t_b][t_hw][i_ckk] = tile_x[t_b][c][t_h + p][t_w + q];
        } else {
          tile_xc[t_b][t_hw][i_ckk] = 0.0;
        }
      }

      for (int i_ckk = t_hw; i_ckk < CKK_TILE_WIDTH; i_ckk += HW_TILE_WIDTH) {
        const int dst_ckk = n * CKK_TILE_WIDTH + i_ckk;
        if ((m < M) && (dst_ckk < n_kernel)) {
          t2d_kt(t_m, i_ckk) = k1d(dst_ckk);
        } else {
          t2d_kt(t_m, i_ckk) = 0.0;
        }
      }

      /*
      dst_ckk = n * TILE_WIDTH + t_m;
      if ((dst_hw < n_hw) && (dst_ckk < n_kernel)) {
        // Do input matrix unroll on-the-fly
        int3 lut = conv_lut[dst_ckk];
        const int q = lut.x, p = lut.y, c = lut.z;
        t2d_xc(t_m, t_hw) = x3d(c, dst_y + p, dst_x + q);
      } else {
        t2d_xc(t_m, t_hw) = 0.0;
      }

      dst_ckk = n * TILE_WIDTH + t_hw;
      if ((m < M) && (dst_ckk < n_kernel)) {
        t2d_kt(t_m, t_hw) = k1d(dst_ckk);
      } else {
        t2d_kt(t_m, t_hw) = 0.0;
      }
      */

      __syncthreads();

      // C_ij = A_ik * B_kj ===> C_ij^T = B_kj^T * A_ik^t
#pragma unroll
      for (int k = 0; k < CKK_TILE_WIDTH; k++) {
        acc += t2d_kt(t_m, k) * t2d_xc(k, t_hw);
      }
      __syncthreads();
    }

    if ((m < M) && (dst_w < W_out) && (dst_h < H_out)) {
      y1d(dst_hw) = acc;
    }
  }

#undef t2d_xc
#undef t2d_kt

#undef y1d
#undef x3d
#undef t2d
#undef k1d
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
  std::cout << "*** constant mem + tiled gemm" << std::endl;
  printf("*** (B=%d, M=%d, C=%d, H=%d, W=%d, K=%d)\n", B, M, C, H, W, K);

  // Estimat output dimension
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  printf("*** (H_out=%d, W_out=%d)\n", H_out, W_out);

  // Calculate needed bytes for original input
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);

#ifdef USE_STREAM
  // Pass through host pointers
  *device_y_ptr = (float*)host_y;
  *device_x_ptr = (float*)host_x;

  // Mark them as pinned memory for asynchronous transfer
  cudaErrChk(cudaHostRegister(*device_y_ptr, bytes_y, cudaHostRegisterPortable));
  cudaErrChk(cudaHostRegister(*device_x_ptr, bytes_x, cudaHostRegisterPortable));
#else
  // Allocate memory on device
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));

  // Copy input data to device
  cudaErrChk(cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
#endif

  // Copy kernel weights
  cudaErrChk(cudaMemcpyToSymbol(kernel, host_k, bytes_k));

  // Calculate lookup table
  int3 host_conv_lut[C * K * K];
  for (int i = 0, c = 0; c < C; c++) {
    for (int p = 0; p < K; p++) {
      for (int q = 0; q < K; q++, i++) {
        host_conv_lut[i] = make_int3(q, p, c);
      }
    }
  }
  const size_t bytes_conv_lut = (C * K * K) * sizeof(int3);
  cudaErrChk(cudaMemcpyToSymbol(conv_lut, host_conv_lut, bytes_conv_lut));
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

/*** Prolog BEGIN ***/
#ifdef USE_STREAM
  // Create streams
  cudaStream_t stream[N_STREAMS];
  for (int i = 0; i < N_STREAMS; i++) {
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
  const int B = ceil((float)B0 / N_STREAMS);
  const int n_y_stream = B * M * H_out * W_out;
  const int n_x_stream = B * C * H * W;

  cudaErrChk(cudaMalloc(&device_y, bytes_y));
  cudaErrChk(cudaMalloc(&device_x, bytes_x));

  // Copy over the relevant data structures to the GPU
  for (int i = 0; i < N_STREAMS; i++) {
    size_t offset = i * n_x_stream;
    size_t bytes = n_x_stream * sizeof(float);
    if (offset + n_x_stream > n_x) {
      // Last stream does not need to copy that much
      bytes = (n_x - offset) * sizeof(float);
    }
    cudaErrChk(cudaMemcpyAsync((void*)&device_x[offset], (void*)&host_x[offset], bytes,
                               cudaMemcpyHostToDevice, stream[i]));
  }
#else   // USE_STREAM
  // Send the entire batch
  const int B = B0;
#endif  // USE_STREAM
  /*** Prolog END ***/

  /*** Kernel call BEGIN ***/
  // Calculate launch size
  dim3 block(HW_TILE_WIDTH, M_TILE_WIDTH, B_BATCH);
  dim3 grid(ceil((float)H_out / TILE_WIDTH) * ceil((float)W_out / TILE_WIDTH),
            ceil((float)M / M_TILE_WIDTH),
            ceil((float)B / B_BATCH));
  printf("*** grid=(x=%d, y=%d, z=%d), block=(x=%d, y=%d, z=%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  // Determine shared memory size
  size_t smem_size =
      B_BATCH * ((HW_TILE_WIDTH + M_TILE_WIDTH) * CKK_TILE_WIDTH + C_MAX * PADDED_TILE_WIDTH * PADDED_TILE_WIDTH) * sizeof(float);
  std::cout << "*** estimated smem.size=" << smem_size / 1024 << "KiB" << std::endl;
  smem_size = 0;  // DEBUG

// Call the kernel
#ifdef USE_STREAM
  for (int i = 0; i < N_STREAMS; i++) {
    size_t offset_y = i * n_y_stream;
    size_t offset_x = i * n_x_stream;
    conv_as_gemm<<<grid, block, smem_size, stream[i]>>>(
        &device_y[offset_y], &device_x[offset_x],
        B, M, C, H, W, K);
  }
#else   // USE_STREAM
  conv_as_gemm<<<grid, block, smem_size>>>(device_y, device_x,
                                           B, M, C, H, W, K);
#endif  // USE_STREAM
  /*** Kernel call END ***/

  /*** Epilog BEGIN ***/
#ifdef USE_STREAM
  // Copy back data to host
  for (int i = 0; i < N_STREAMS; i++) {
    size_t offset = i * n_y_stream;
    size_t bytes = n_y_stream * sizeof(float);
    if (offset + n_y_stream > n_y) {
      // Last stream does not need to copy that much
      bytes = (n_y - offset) * sizeof(float);
    }
    cudaErrChk(cudaMemcpyAsync(&host_y[offset], &device_y[offset], bytes,
                               cudaMemcpyDeviceToHost, stream[i]));
  }

  // Destory streams
  for (int i = 0; i < N_STREAMS; i++) {
    cudaErrChk(cudaStreamDestroy(stream[i]));
  }
#else   // USE_STREAM
  // nop
#endif  // USE_STREAM
  /*** Epilog END ***/

  // We directly wait for the single kernel to end
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
#ifdef USE_STREAM
  // Data is already write back to host earlier, safe to clean up now

  // Release pinned memory
  cudaErrChk(cudaHostUnregister(device_y));
  cudaErrChk(cudaHostUnregister(device_x));
#else   // USE_STREAM
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);

  // Copy output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_x));
#endif  // USE_STREAM
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
