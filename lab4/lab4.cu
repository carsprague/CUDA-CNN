#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_SIZE 8 
//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float subTile[TILE_SIZE+MASK_WIDTH-1][TILE_SIZE+MASK_WIDTH-1][TILE_SIZE+MASK_WIDTH-1];

  // note that threadIdx is relative to the tile, not the entire input
  // e.g. 0 -> TILE_SIZE+(MASK_WIDTH-1)
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  int radius = MASK_WIDTH / 2;

  int row_o = blockIdx.y * TILE_SIZE + ty;
  int col_o = blockIdx.x * TILE_SIZE + tx;
  int dep_o = blockIdx.z * TILE_SIZE + tz;
  int row_i = row_o - radius;
  int col_i = col_o - radius;
  int dep_i = dep_o - radius;
  // width = x_size, height = y_size, depth = z_size
  if ((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (dep_i >= 0) && (dep_i < z_size)) {
    subTile[tz][ty][tx] = input[dep_i*y_size*x_size+row_i*x_size+col_i];
  } else {
    subTile[tz][ty][tx] = 0.0;
  }
  __syncthreads();

  float pvalue = 0;
  if (tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE) {
    for (int i = 0; i < MASK_WIDTH; ++i) {
      for (int j = 0; j < MASK_WIDTH; ++j) {
        for (int k = 0; k < MASK_WIDTH; ++k) {
          pvalue += subTile[tz+i][ty+j][tx+k] * M[i][j][k];
        }
      }
    }
    if (row_o < y_size && col_o < x_size && dep_o < z_size) {
      output[dep_o*y_size*x_size+row_o*x_size+col_o] = pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**)&deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void**)&deviceOutput, (inputLength-3) * sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, kernelLength * sizeof(float));

  //@@ Initialize grid and block dimensions here
  dim3 DimBlock(TILE_SIZE+(MASK_WIDTH-1),TILE_SIZE+(MASK_WIDTH-1),TILE_SIZE+(MASK_WIDTH-1));
  dim3 DimGrid(ceil(x_size/(1.0*TILE_SIZE)),
               ceil(y_size/(1.0*TILE_SIZE)),
               ceil(z_size/(1.0*TILE_SIZE)));
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

