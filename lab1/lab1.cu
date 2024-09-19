// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  // note that cudaMalloc does not return a pointer, but rather returns cudaError_t
  // cudaMalloc(void** devPtr, size_t size)
  cudaError_t err;
  err = cudaMalloc((void**)&deviceInput1, inputLength * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMalloc deviceInput1\n");
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&deviceInput2, inputLength * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMalloc deviceInput2\n");
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMalloc deviceOutput\n");
    exit(EXIT_FAILURE);
  }

  //@@ Copy memory to the GPU here
  // cudaMemcpy(void* dest, const void* src, size_t count, cudaMemcpyKind kind)
  err = cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMemcpy deviceInput1\n");
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMemcpy deviceInput2\n");
    exit(EXIT_FAILURE);
  }

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength * sizeof(float)/256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  err = cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed cudaMemcpy hostOutput\n");
    exit(EXIT_FAILURE);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
