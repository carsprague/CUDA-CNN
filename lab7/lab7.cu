// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE       128

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)                          

//@@ insert code here
__global__ void float_to_char(float *input, unsigned char *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx;
  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      idx = (y*width+x)*3 + c;
      output[idx] = (unsigned char)(input[idx] * 255);
    }
  }  
}

__global__ void rgb_to_gray(unsigned char *image, unsigned char *grayImage, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned char r, g, b;
  int idx = y*width+x;
  if (idx < width*height) {
    r = image[3*idx];
    g = image[3*idx+1];
    b = image[3*idx+2];
    grayImage[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void hist(unsigned *hist, unsigned char *image, int width, int height) {
  __shared__ unsigned priv[256];
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y*width+x;
  int t = threadIdx.x + threadIdx.y * blockDim.x; // unique idx in blk
  if (t < 256) priv[t] = 0;
  __syncthreads();
  if (x < width && y < height) atomicAdd(&(priv[image[idx]]), 1);
  __syncthreads();
  if (t < 256) atomicAdd(&(hist[t]), priv[t]);
}

__global__ void scan(unsigned int *input, float *output, int width, int height) {
  __shared__ float T[2*BLOCK_SIZE];
  float pixels = (float)(width * height);

  // load T
  unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 2*BLOCK_SIZE) {
    T[threadIdx.x]  = input[i] / pixels;
    // since the kernel runs by blocks, we have to manually add the second set of elems to T
    if (i + BLOCK_SIZE < 2*BLOCK_SIZE) T[threadIdx.x + BLOCK_SIZE] = input[i + BLOCK_SIZE] / pixels;
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
  if (i < 2*BLOCK_SIZE) {
    output[i] = T[threadIdx.x];
    if (i + BLOCK_SIZE < 2*BLOCK_SIZE) output[i + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];
  }
}

__global__ void apply_equalization(unsigned char* image, float* cdf, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  float cc;
  int idx;

  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      idx = (y*width+x)*3 + c;
      cc = (float)(min(max(255*(cdf[image[idx]] - cdf[0])/(1.0 - cdf[0]), 0.0), 255.0));
      image[idx] = (unsigned char)cc;
    }
  }
}

__global__ void char_to_float(unsigned char *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx;

  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      idx = (y*width+x)*3 + c;
      output[idx] = (float)(input[idx] / 255.0);
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char* charImage;
  unsigned char* grayCharImage;
  unsigned int *histogram;
  float *fhistogram;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  /* debugging */
  unsigned char* hostCharImage;
  unsigned char* hostGrayCharImage;
  unsigned int* hostHistogram;
  float* hostFhistogram;
  hostCharImage = (unsigned char*)malloc(imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  hostGrayCharImage = (unsigned char*)malloc(imageWidth*imageHeight*sizeof(unsigned char));
  hostHistogram = (unsigned int*)malloc(256*sizeof(unsigned int));
  hostFhistogram = (float*)malloc(256*sizeof(float));
  /* setup stuff  */
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  /* holy cow thats a lot of variables */
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float)));
  wbCheck(cudaMalloc((void **)&charImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&grayCharImage, imageWidth*imageHeight*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH*sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&fhistogram, HISTOGRAM_LENGTH*sizeof(float)));

  wbCheck(cudaMemset(histogram, 0, HISTOGRAM_LENGTH*sizeof(int)));
  wbCheck(cudaMemset(deviceOutputImageData, 0, imageWidth*imageHeight*imageChannels*sizeof(float)));
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice));
  //@@ insert code here - kernel calls & such
  dim3 DimBlockIm(32, 32, 1);
  dim3 DimGridIm((ceil(imageWidth/32.0)), (ceil(imageHeight/32.0)), 1);
  dim3 DimBlockSc(256, 1, 1);
  dim3 DimGridSc(1, 1, 1);

  ///////////////////////////
  printf("INITIAL DATA: ");
  for (int i = 0; i < 20; i++) {
    printf("%f ", hostInputImageData[i]);
  }
  printf("\n");
  ///////////////////////////

  ///////////////////////////
  float_to_char<<<DimGridIm, DimBlockIm>>>(deviceInputImageData, charImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostCharImage, charImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  printf("AFTER CONVERT TO CHAR: ");
  for (int i = 0; i < 20; i++) {
    printf("%d ", hostCharImage[i]);
  }
  printf("\n");
  ///////////////////////////

  ///////////////////////////
  rgb_to_gray<<<DimGridIm, DimBlockIm>>>(charImage, grayCharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostGrayCharImage, grayCharImage, imageWidth*imageHeight*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  printf("AFTER CONVERT TO GRAY: ");
  for (int i = 0; i < 20; i++) {
    printf("%d ", hostGrayCharImage[i]);
  }
  printf("\n");
  ///////////////////////////

  ///////////////////////////
  hist<<<DimGridIm, DimBlockIm>>>(histogram, grayCharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostHistogram, histogram, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("HISTOGRAM: ");
  for (int i = 0; i < 256; i++) {
    printf("%d,", hostHistogram[i]);
  }
  printf("\n");
  ///////////////////////////

  ///////////////////////////
  scan<<<DimGridSc, DimBlockSc>>>(histogram, fhistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostFhistogram, fhistogram, 256*sizeof(float), cudaMemcpyDeviceToHost));
  printf("SCANNED FHISTOGRAM: ");
  for (int i = 0; i < 256; i++) {
    printf("%f,", hostFhistogram[i]);
  }
  printf("\n");
  ///////////////////////////

  ///////////////////////////
  apply_equalization<<<DimGridIm, DimBlockIm>>>(charImage, fhistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  ///////////////////////////
  char_to_float<<<DimGridIm, DimBlockIm>>>(charImage, deviceOutputImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost));
  printf("MY FINAL DATA: ");
  for (int i = 0; i < 20; i++) {
    printf("%f ", hostOutputImageData[i]);
  }
  printf("\n");
  ///////////////////////////

  /* from the campuswire #371: 'If you’ve copied deviceOutputImage to hostOutputImage correctly, you can use:' */
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(histogram);
  cudaFree(charImage);
  cudaFree(grayCharImage);
  free(hostInputImageData);
  free(hostOutputImageData);
  free(hostCharImage);
  free(hostGrayCharImage);
  free(hostHistogram);

  printf("\n");

  return 0;
}

