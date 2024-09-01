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

/*---------------------------------Constant Memory-----------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_SIZE  16

// //__constant__ float M[Channel][Map_out][K][K];
// __constant__ float M[10000];

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) M[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
//     //int H_grid = Height_out / TILE_SIZE;
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
//     int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
//     if (h < Height_out && w < Width_out) {
//         float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
//                 }
//             }
//         }
//         out_4d(n, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

//     /* allocated memory with a double pointer */
//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

//     /* copy memory using a single pointer */
//     cudaMemcpyToSymbol(M, host_mask, Channel*Map_out*K*K*sizeof(float));
//     //cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_grid = ceil(W_out/(TILE_SIZE*1.0));
//     int H_grid = ceil(H_out/(TILE_SIZE*1.0));
//     int Z = H_grid * W_grid;
//     dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 GridDim(Batch, Map_out, Z);
//     conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }

/*---------------------------------FP16---------------------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "cuda_fp16.h"
// #include "gpu-new-forward.h"

// #define TILE_SIZE  16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
//     //int H_grid = Height_out / TILE_SIZE;
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
//     int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
//     if (h < Height_out && w < Width_out) {
//         __half2 acc = __float2half2_rn(0.0f);
//         //float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     __half2 in = __floats2half2_rn(in_4d(n*2, c, h+p, w+q), in_4d(2*n+1, c, h+p, w+q));
//                     __half2 ma = __float2half2_rn(mask_4d(m, c, p, q));
//                     acc = __hfma2(in, ma, acc);
//                     //acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
//                 }
//             }
//         }
//         float2 out = __half22float2(acc);
//         out_4d(2*n, m, h, w) = out.x;
//         out_4d(2*n+1, m, h, w) = out.y;
//         //out_4d(n, m, h, w+1) = out.y;
//         //out_4d(n, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

//     /* allocated memory with a double pointer */
//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

//     /* copy memory using a single pointer */
//     cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_grid = ceil(W_out/(TILE_SIZE*1.0));
//     int H_grid = ceil(H_out/(TILE_SIZE*1.0));
//     int Z = H_grid * W_grid;
//     dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 GridDim(Batch/2, Map_out, Z);
//     conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }


/*----------------------------------Restrict---------------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_SIZE  16

// __global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
//     //int H_grid = Height_out / TILE_SIZE;
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
//     int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
//     if (h < Height_out && w < Width_out) {
//         float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             #pragma unroll 7
//             for (int p = 0; p < K; p++) {
//                 #pragma unroll 7
//                 for (int q = 0; q < K; q++) {
//                     acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
//                 }
//             }
//         }
//         out_4d(n, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

//     /* allocated memory with a double pointer */
//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

//     /* copy memory using a single pointer */
//     cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_grid = ceil(W_out/(TILE_SIZE*1.0));
//     int H_grid = ceil(H_out/(TILE_SIZE*1.0));
//     int Z = H_grid * W_grid;
//     dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 GridDim(Batch, Map_out, Z);
//     conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
//     //printf("K: %d, Channel: %d\n", K, Channel);
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }

/*-------------------------------------------------Unrolling-----------------------------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 32

// __global__ void unroll_k(int C, int H, int W, int K, const float *X, float *X_unroll) {
//     #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define unroll_out(i2, i1, i0) X_unroll[(i2) * (C*K*K*H_out*W_out) + (i1) * (H_out*W_out) + (i0)]

//     int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
//     int t = blockIdx.x * blockDim.x + threadIdx.x;
//     int b = blockIdx.y;
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;
//     int W_unroll = H_out * W_out;

//     if (t < C * W_unroll) {
//         c = t / W_unroll;
//         s = t % W_unroll;
//         h_out = s / W_out;
//         w_out = s % W_out;
//         h_unroll = h_out * W_out + w_out;
//         w_base = c * K * K;
//         for (p = 0; p < K; p++) {
//             for (q = 0; q < K; q++) {
//                 w_unroll = w_base + p * K + q;
//                 unroll_out(b, w_unroll, h_unroll) = in_4d(b, c, h_out+p, w_out+q);
//             }
//         }
//     }

//     #undef in_4d
//     #undef unroll_out
// }


// __global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
//                                      int numARows, int numAColumns,
//                                      int numBRows, int numBColumns,
//                                      int numCRows, int numCColumns, int W_out) {

//   // load the memory into the shared section (way faster than global)
//   __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;

//   float pvalue = 0;
//     for (int k = 0; k < (numAColumns-1)/TILE_WIDTH+1; ++k) {
//         if (row < numARows && k*TILE_WIDTH+tx < numAColumns) {
//         subTileM[ty][tx] = A[row*numAColumns+k*TILE_WIDTH+tx];
//         } else {
//         subTileM[ty][tx] = 0;
//         }
//         if (k*TILE_WIDTH+ty < numBRows && col < numBColumns) {
//         subTileN[ty][tx] = B[blockIdx.z*(numAColumns*numCColumns)+(k*TILE_WIDTH+ty)*numBColumns+col];
//         } else {
//         subTileN[ty][tx] = 0;
//         }
//         __syncthreads();
//         for (int q = 0; q < TILE_WIDTH; ++q) {
//         pvalue += subTileM[ty][q] * subTileN[q][tx];
//         }
//         __syncthreads();
//     }
//     if (row < numCRows && col < numCColumns) {
//         C[blockIdx.z*numCRows*numCColumns+row*numCColumns+(col/W_out)*W_out+(col%W_out)] = pvalue;
//     }
// }
// float *device_unrolled;
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {

//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));
//     cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);

//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_unroll = Channel * K * K;
//     int H_unroll = H_out * W_out;

//     cudaMalloc((void**)&device_unrolled, (Batch/4)*W_unroll*H_unroll*sizeof(float));
//     cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int numARows = Map_out;
//     int numAColumns = Channel*K*K;
//     int numBRows = Channel*K*K;
//     int numBColumns =  W_out*H_out;
//     int numCRows = numARows;
//     int numCColumns = numBColumns;
    
//     dim3 DimBlockU(1024, 1, 1);
//     dim3 DimGridU(ceil((Channel*H_out*W_out)/1024.0), Batch/4, 1);
//     dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     dim3 DimGrid(ceil(1.0*numBColumns/(float)TILE_WIDTH), ceil(1.0*numARows/(float)TILE_WIDTH), Batch/4);

//     for (int i = 0; i < 4; i++) {
//         unroll_k<<<DimGridU, DimBlockU>>>(Channel, Height, Width, K, device_input+i*(Batch/4)*Height*Width*Channel, device_unrolled);
        
//         cudaDeviceSynchronize();

//         matrixMultiplyShared<<<DimGrid, DimBlock>>>(device_mask, device_unrolled, device_output+i*((Batch/4)*(Height-K+1)*(Width-K+1)*Map_out),
//                                                     numARows, numAColumns,
//                                                     numBRows, numBColumns,
//                                                     numCRows, numCColumns, W_out);

//         cudaDeviceSynchronize();

//     }
//     return;
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }

/*-----------------------------------Streams-------------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_SIZE  16
// //#define NUM_STREAMS 100

// /* used in streams */
// const float *shost_input;
// const float *shost_output;

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
//     //int H_grid = Height_out / TILE_SIZE;
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
//     int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
//     if (h < Height_out && w < Width_out) {
//         float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
//                 }
//             }
//         }
//         out_4d(n, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// __host__ void stream_execute(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K);
	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

    
//     /* allocated memory with a double pointer */
//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

//     /* workaround so we have access to the host data */
//     shost_input = host_input;
//     shost_output = host_output;

//     /* copy memory using a single pointer */
//     cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaHostRegister((void*)host_input, Batch*Height*Width*Channel*sizeof(float), cudaHostRegisterDefault);
//     cudaHostRegister((void*)host_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaHostRegisterDefault);

//     stream_execute(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
// }

// /* this is a workaround function so we can do all the memcpying and kernel launching with streams */
// __host__ void stream_execute(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_grid = ceil(W_out/(TILE_SIZE*1.0));
//     int H_grid = ceil(H_out/(TILE_SIZE*1.0));
//     int Z = H_grid * W_grid;
//     int NUM_STREAMS = 10;
//     dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 GridDim(Batch/NUM_STREAMS, Map_out, Z);
//     cudaStream_t streams[NUM_STREAMS];

//     int sinput_size = (Batch*Height*Width*Channel)/NUM_STREAMS;
//     int soutput_size = (Batch*(Height-K+1)*(Width-K+1)*Map_out)/NUM_STREAMS;

//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaStreamCreate(&streams[i]);
//         cudaMemcpyAsync((void*)(device_input + i*sinput_size), shost_input+(i*sinput_size), sinput_size*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//         //cudaStreamSynchronize(streams[i]);
//         conv_forward_kernel<<<GridDim, BlockDim, 0, streams[i]>>>(device_output+(i*soutput_size), device_input+(i*sinput_size), device_mask, Batch, Map_out, Channel, Height, Width, K);
//         //cudaStreamSynchronize(streams[i]);
//         cudaMemcpyAsync((void*)(shost_output+(i*soutput_size)), (device_output + i*soutput_size), soutput_size*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
//         //cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }
//     cudaDeviceSynchronize();
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* And since con_forward_gpu host function does not have the host vectors to copy over to the device, you'll need to shift your host side implementation to a host function which does have those host vectors. - campuswire #512*/
//     return;
// }

// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     cudaFreeHost((void*)shost_input);
//     cudaFreeHost((void*)shost_output);
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }

/*----------------------------------Baseline--------------------------------------*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_SIZE  16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil(Width_out/(TILE_SIZE*1.0));
//     //int H_grid = Height_out / TILE_SIZE;
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
//     int w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
//     if (h < Height_out && w < Width_out) {
//         float acc = 0.0f;
//         for (int c = 0; c < Channel; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     acc += in_4d(n, c, h+p, w+q) * mask_4d(m, c, p, q);
//                 }
//             }
//         }
//         out_4d(n, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

//     /* allocated memory with a double pointer */
//     cudaMalloc((void**)device_mask_ptr, Channel*Map_out*K*K*sizeof(float));
//     cudaMalloc((void**)device_input_ptr, Batch*Height*Width*Channel*sizeof(float));
//     cudaMalloc((void**)device_output_ptr, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float));

//     /* copy memory using a single pointer */
//     cudaMemcpy(*device_mask_ptr, host_mask, Channel*Map_out*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, Batch*Height*Width*Channel*sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;
//     int W_grid = ceil(W_out/(TILE_SIZE*1.0));
//     int H_grid = ceil(H_out/(TILE_SIZE*1.0));
//     int Z = H_grid * W_grid;
//     dim3 BlockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 GridDim(Batch, Map_out, Z);
//     conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch*(Height-K+1)*(Width-K+1)*Map_out*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }