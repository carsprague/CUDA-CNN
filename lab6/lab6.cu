// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float* sum, int len, int load_sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  // load T
  unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    T[threadIdx.x]  = input[i];
    // since the kernel runs by blocks, we have to manually add the second set of elems to T
    if (i + BLOCK_SIZE < len) T[threadIdx.x + BLOCK_SIZE] = input[i + BLOCK_SIZE];
    else T[threadIdx.x + BLOCK_SIZE] = 0.0;
  } else {
    T[threadIdx.x] = 0.0;
  }
  
  // reduction step
  int stride = 1;
  while (stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2-1;
    if (idx < 2*BLOCK_SIZE && (idx - stride) >= 0) {
      T[idx] += T[idx - stride];
    }
    stride = stride*2;
  }

  // distribution tree
  stride = BLOCK_SIZE/2;
  while (stride > 0) {
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2-1;
    if ((idx+stride) < 2*BLOCK_SIZE) {
      T[idx+stride] += T[idx];
    }
    stride = stride/2;
  }

  // load output array
  __syncthreads();
  if (i < len) {
    output[i] = T[threadIdx.x];
    if (i + BLOCK_SIZE < len) output[i + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];
  }

  // load the auxillary array in the first kernel call (only last thread does it)
  __syncthreads();
  if (load_sum && threadIdx.x == blockDim.x-1) sum[blockIdx.x] = T[2*BLOCK_SIZE-1];
}

// note that this is called with a blockDim of 2*BLOCK_SIZE
// meaning do one operation per thread, heavily inspired by
// chapter 8 in the book
__global__ void sum(float *input, float *sum, int len) {
  if (!blockIdx.x) return; // out of bounds check
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) input[i] += sum[blockIdx.x-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  // used for sum
  dim3 DimBlockSum(BLOCK_SIZE*2, 1, 1);
  // T is of size BLOCK_SIZE*2
  dim3 DimGrid(ceil(numElements/((float)(BLOCK_SIZE*2))), 1, 1);
  // after we write to the auxillary array we only need to launch one block
  dim3 DimGridSolo(1, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float* deviceAux;
  cudaMalloc((void **)&deviceAux, 2*BLOCK_SIZE*sizeof(float));
  int TRUE = 1, FALSE = 0;

  // heavily inspired by chapter 8 of the book
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, deviceAux, numElements, TRUE);
  cudaDeviceSynchronize();
  scan<<<DimGridSolo, DimBlock>>>(deviceAux, deviceAux, NULL, ceil(numElements/((float)(BLOCK_SIZE*2))), FALSE);
  cudaDeviceSynchronize();
  sum<<<DimGrid, DimBlockSum>>>(deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

