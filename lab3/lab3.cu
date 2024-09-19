#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP

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
      subTileN[ty][tx] = B[(k*TILE_WIDTH+ty)*numBColumns+col];
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
    C[row*numCColumns+col] = pvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  //@@ Allocate GPU memory here
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  cudaMalloc((void**)&deviceInput1, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**)&deviceInput2, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**)&deviceOutput, numCRows * numCColumns * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  // HUGELY IMPORTANT, CAST TILE_WIDTH TO FLOAT
  dim3 DimGrid(ceil(1.0*numBColumns/(float)TILE_WIDTH), ceil(1.0*numARows/(float)TILE_WIDTH), 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceOutput, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
