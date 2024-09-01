#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

__global__ void unroll_k(int C, int H, int W, int K, const float *X, float *X_unroll) {
    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define unroll_out(i2, i1, i0) X_unroll[(i2) * (C*K*K*H_out*W_out) + (i1) * (H_out*W_out) + (i0)]

    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                w_unroll = w_base + p * K + q;
                unroll_out(b, w_unroll, h_unroll) = in_4d(b, c, h_out+p, w_out+q);
            }
        }
    }

    #undef in_4d
    #undef unroll_out
}


__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int W_out) {

  // load the memory into the shared section (way faster than global)
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float pvalue = 0;
    for (int k = 0; k < (numAColumns-1)/TILE_WIDTH+1; ++k) {
        if (row < numARows && k*TILE_WIDTH+tx < numAColumns) {
        subTileM[ty][tx] = A[row*numAColumns+k*TILE_WIDTH+tx];
        } else {
        subTileM[ty][tx] = 0;
        }
        if (k*TILE_WIDTH+ty < numBRows && col < numBColumns) {
        subTileN[ty][tx] = B[blockIdx.z*(numAColumns*numCColumns)+(k*TILE_WIDTH+ty)*numBColumns+col];
        } else {
        subTileN[ty][tx] = 0;
        }
        __syncthreads();
        for (int q = 0; q < TILE_WIDTH; ++q) {
        pvalue += subTileM[ty][q] * subTileN[q][tx];
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns) {
        C[blockIdx.z*numCRows*numCColumns+row*numCColumns+(col/W_out)*W_out+(col%W_out)] = pvalue;
    }
}
float *device_unrolled;
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);

    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int W_unroll = Channel * K * K;
    int H_unroll = H_out * W_out;

    cudaMalloc((void**)&device_unrolled, (Batch/4)*W_unroll*H_unroll*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int numARows = Map_out;
    int numAColumns = Channel*K*K;
    int numBRows = Channel*K*K;
    int numBColumns =  W_out*H_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;
    
    dim3 DimBlockU(1024, 1, 1);
    dim3 DimGridU(ceil((Channel*H_out*W_out)/1024.0), Batch/4, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(ceil(1.0*numBColumns/(float)TILE_WIDTH), ceil(1.0*numARows/(float)TILE_WIDTH), Batch/4);

    for (int i = 0; i < 4; i++) {
        unroll_k<<<DimGridU, DimBlockU>>>(Channel, Height, Width, K, device_input+i*(Batch/4)*Height*Width*Channel, device_unrolled);
        
        cudaDeviceSynchronize();

        matrixMultiplyShared<<<DimGrid, DimBlock>>>(device_mask, device_unrolled, device_output+i*((Batch/4)*(Height-K+1)*(Width-K+1)*Map_out),
                                                    numARows, numAColumns,
                                                    numBRows, numBColumns,
                                                    numCRows, numCColumns, W_out);

        cudaDeviceSynchronize();

    }
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_mask);
    cudaFree(device_input);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}