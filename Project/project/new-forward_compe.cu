/*---------------------------------Final Submission---------------------------------*/
#include <cmath>
#include <iostream>
#include "cuda_fp16.h"
#include "gpu-new-forward.h"

#define TILE_SIZE  16

__constant__ float M[10000];

__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) M[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
    //int H_grid = Height_out / TILE_SIZE;
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
    if (h < Height_out && w < Width_out) {
        __half2 acc = __float2half2_rn(0.0f);
        //float acc = 0.0f;
        for (int c = 0; c < Channel; c++) {
            #pragma unroll 7
            for (int p = 0; p < K; p++) {
                #pragma unroll 7
                for (int q = 0; q < K; q++) {
                    __half2 in = __floats2half2_rn(in_4d(n*2, c, h+p, w+q), in_4d(2*n+1, c, h+p, w+q));
                    __half2 ma = __float2half2_rn(mask_4d(m, c, p, q));
                    acc = __hfma2(in, ma, acc);
                    //acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
                }
            }
        }
        float2 out = __half22float2(acc);
        out_4d(2*n, m, h, w) = out.x;
        out_4d(2*n+1, m, h, w) = out.y;
        //out_4d(n, m, h, w+1) = out.y;
        //out_4d(n, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    /* allocated memory with a double pointer */
    cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

    /* copy memory using a single pointer */
    //cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, host_mask, Channel*Map_out*K*K*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int W_grid = ceil(W_out/(TILE_SIZE*1.0));
    int H_grid = ceil(H_out/(TILE_SIZE*1.0));
    int Z = H_grid * W_grid;
    dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 GridDim(Batch/2, Map_out, Z);
    conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
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