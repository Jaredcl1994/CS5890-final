#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <cuda.h>

__device__ __constant__ int HEIGHT;
__device__ __constant__ int WIDTH;
__device__ __constant__ int BLOCK_WIDTH;
__device__ double d_max;

// check for an error on cuda functions
inline cudaError_t checkCuda(cudaError_t result, int s) {
    if (result != cudaSuccess) {
        fprintf(stderr, "error: %d, %s\n", s, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void assertEqual(unsigned char* img1, unsigned char* img2, int n);

void serialCanny(unsigned char* colorImage, unsigned char* finalImage, int width, int height);
void serialGrayscale(unsigned char* colorImage, unsigned char* grayImage, int width, int height);
void serialGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage, int width, int height);
void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions, int width, int height);
void serialNonMaxSuppression(unsigned char* edgeImage, unsigned char* suppressedImage, float* directions, int width, int height);
void serialThreshold(unsigned char* suppressedImage, unsigned char* thresholdImage, int width, int height);
void serialCleanup(unsigned char* thresholdImage, unsigned char* finalImage, int width, int height);

__global__ void gpuCanny(unsigned char* colorImage, unsigned char* finalImage);
__global__ void gpuGrayscale(unsigned char* colorImage, unsigned char* grayImage);
__global__ void gpuGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage);
__global__ void gpuEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions);
__global__ void gpuNonMaxSuppression(unsigned char* edgeImage, unsigned char* suppressedImage, float* directions);
__global__ void gpuThreshold(unsigned char* suppressedImage, unsigned char* thresholdImage);
__global__ void gpuCleanup(unsigned char* thresholdImage, unsigned char* finalImage);

int main() {
  double max = 0;
  int width = 3379;
  int height = 3005;
  int block_width = 16;
  checkCuda(cudaMemcpyToSymbol(HEIGHT, &height, 1*sizeof(int)), 2);
  checkCuda(cudaMemcpyToSymbol(WIDTH, &width, 1*sizeof(int)), 3);
  checkCuda(cudaMemcpyToSymbol(BLOCK_WIDTH, &block_width, 1*sizeof(int)), 4);

  int n = width*height*3;
  unsigned char *d_original;
  unsigned char *d_grayscale;
  unsigned char *d_blurred;
  unsigned char *d_edge;
  float *d_directions;
  // unsigned char *d_suppressed;
  // unsigned char *d_threshold;
  // unsigned char *d_final;
  unsigned char *originalImage = (unsigned char*) malloc(n * sizeof(unsigned char));
  unsigned char *grayscaleImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *baselineGrayscale = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *blurredImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *baselineBlurred = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *edgeImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *baselineEdge = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  float *directions = (float*) malloc(width*height * sizeof(float));
  float *baselineDirections = (float*) malloc(width*height * sizeof(float));
  // unsigned char *suppressedImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  // unsigned char *baselineSuppressed = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  // unsigned char *thresholdImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  // unsigned char *baselineThreshold = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  // unsigned char *finalImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  // unsigned char *baselineFinal = (unsigned char*) malloc(width*height * sizeof(unsigned char));

  // read original image
  FILE *imageFile;
  imageFile = fopen("butterfly_nebula_3379x3005_rgb.raw", "rb");
  fread(originalImage, sizeof(unsigned char), n, imageFile);
  fclose(imageFile);

  // calculate baseline
  // serialCanny(originalImage, baselineFinal, width, height);
  serialGrayscale(originalImage, baselineGrayscale, width, height);
  serialGaussianBlur(baselineGrayscale, baselineBlurred, width, height);
  serialEdgeDetection(baselineBlurred, baselineEdge, baselineDirections, width, height);
  // serialNonMaxSuppression(baselineEdge, baselineSuppressed, baselineDirections, width, height);
  // serialThreshold(baselineSuppressed, baselineThreshold, width, height);
  // serialCleanup(baselineThreshold, baselineFinal, width, height);

  // serial canny edge detection
  // serialCanny(originalImage, finalImage, width, height); // COMPLETE

  // copy data arrays
  checkCuda(cudaMalloc((void**)&d_original, n*sizeof(unsigned char)), 5);
  checkCuda(cudaMalloc((void**)&d_grayscale, width*height*sizeof(unsigned char)), 6);
  checkCuda(cudaMalloc((void**)&d_blurred, width*height*sizeof(unsigned char)), 69);
  checkCuda(cudaMalloc((void**)&d_edge, width*height*sizeof(unsigned char)), 68);
  checkCuda(cudaMalloc((void**)&d_directions, width*height*sizeof(float)), 70);
  checkCuda(cudaMemcpy(d_original, originalImage, n*sizeof(unsigned char), cudaMemcpyHostToDevice), 7);
  checkCuda(cudaMemcpyToSymbol(d_max, &max, 1*sizeof(double)), 40);

  // Launch GPU kernel
  dim3 dimGrid(ceil((1.0*width)/block_width), ceil((1.0*width)/block_width), 1);
  dim3 dimBlock(block_width, block_width, 1);
  gpuGrayscale<<<dimGrid, dimBlock>>>(d_original, d_grayscale);
  gpuGaussianBlur<<<dimGrid, dimBlock>>>(d_grayscale, d_blurred);
  gpuEdgeDetection<<<dimGrid, dimBlock>>>(d_blurred, d_edge, d_directions);

  // gpu canny edge detection (different file)
  checkCuda(cudaMemcpy(grayscaleImage, d_grayscale, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost), 8);
  checkCuda(cudaMemcpy(blurredImage, d_blurred, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost), 80);
  checkCuda(cudaMemcpy(edgeImage, d_edge, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost), 800);
  checkCuda(cudaFree(d_original), 9);
  checkCuda(cudaFree(d_grayscale), 10);
  checkCuda(cudaFree(d_blurred), 20);
  checkCuda(cudaFree(d_edge), 200);

  assertEqual(grayscaleImage, baselineGrayscale, width*height); // passed
  assertEqual(blurredImage, baselineBlurred, width*height);
  assertEqual(edgeImage, baselineEdge, width*height);
  // assertEqual(suppressedImage, baselineSuppressed, width*height);
  // assertEqual(thresholdImage, baselineThreshold, width*height);
  // assertEqual(finalImage, baselineFinal, width*height);

  // outputFile = fopen("finalImage.raw", "wb+");
  // fwrite(finalImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//*******************************GPU VERSION************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void gpuCanny(unsigned char* originalImage, unsigned char* finalImage, int width, int height) {
  unsigned char *grayscaleImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *blurredImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *edgeImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  float *directions = (float*) malloc(width*height * sizeof(float));
  unsigned char *suppressedImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *thresholdImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));

 // gpuGrayscale(originalImage, grayscaleImage);
 // gpuGaussianBlur(grayscaleImage, blurredImage);
 // gpuEdgeDetection(blurredImage, edgeImage, directions, max);
 // gpuNonMaxSuppression(edgeImage, suppressedImage, directions);
 // gpuThreshold(suppressedImage, thresholdImage);
 // gpuCleanup(thresholdImage, finalImage);

}

__global__ void gpuGrayscale(unsigned char* colorImage, unsigned char* grayImage) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // if it's within bounds, assign it to its transposed counterpart
  if (col < WIDTH && row < HEIGHT) {
    int i = 3 * (row * WIDTH + col);
    int j = row * WIDTH + col;

    unsigned char temp = (colorImage[i] + colorImage[i+1] +  colorImage[i+2])/3;
    grayImage[j] = temp;
  }
}

__global__ void gpuGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage) {
  unsigned char val = 0;
  int offset, row, col, i, r, c;
  float gauss[3][3] = {{0.01, 0.08, 0.01}, {0.08, 0.64, 0.08}, {0.01, 0.08, 0.01}};
  col = threadIdx.x + blockIdx.x * blockDim.x;
  row = threadIdx.y + blockIdx.y * blockDim.y;
  i = row * WIDTH + col;

  if (col < WIDTH && row < HEIGHT) {
    for (r=-1; r<2; r++) {
      for (c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=HEIGHT-1 && r > 0) || (col>=WIDTH-1 && c > 0)) {
          val += 0;
        } else {
          // printf("row: %d, col: %d, ", r, c);
          offset = i+r*WIDTH+c;
          val += grayImage[offset]*gauss[r+1][c+1];
        }
      }
    }
    blurredImage[row*WIDTH+col] = val;
  }
}

__global__ void gpuEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions) {
  double valx, valy, temp;
  int offset, row, col, i, r, c;
  int kx[3][3] = {{-1, 0, 1}, {-2, 0, -2}, {-1, 0, 1}};
  int ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
  col = threadIdx.x + blockIdx.x * blockDim.x;
  row = threadIdx.y + blockIdx.y * blockDim.y;

  i = row * WIDTH + col;
  if (col < WIDTH && row < HEIGHT) {
    valx= 0;
    valy= 0;
    for (r=-1; r<2; r++) {
      for (c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=HEIGHT-1 && r > 0) || (col>=WIDTH-1 && c > 0)) {
          valx+=0;
          valy+=0;
        } else {
          offset = i+r*WIDTH+c;
          valx += (double)blurredImage[offset]*kx[r+1][c+1];
          valy += (double)blurredImage[offset]*ky[r+1][c+1];
        }
      }
    }
    temp = sqrt(pow(valx, 2.0) + pow(valy, 2.0));
    if (temp > d_max) d_max = temp;
    edgeImage[i] = temp;
    directions[i] = atan(valy/valx);
  }

  __syncthreads();

  if (col < WIDTH && row < HEIGHT) {
    edgeImage[i] = edgeImage[i]/d_max*255;
  }
}

__global__ void gpuNonMaxSuppression(unsigned char* edgeImage, unsigned char* suppressedImage, float* directions) {
  unsigned char val1, val2;
  int row, col, i, r, c;
  float angle;
  for (i=0; i<WIDTH*HEIGHT; i++) {
    val1 = 255;
    val2 = 255;
    row = i/WIDTH;
    col = i%WIDTH;
    angle = directions[i] * 180 / 3.141592653;
    if (angle < 0) angle += 180;

    for (r=-1; r<2; r++) {
      for (c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row==HEIGHT-1 && r > 0) || (col==WIDTH-1 && c > 0)) {
          ;
        } else {
          // 0 angle
          if ((r==0 && c==1) && (0 <= angle && angle < 22.5 || 157.5 <= angle <= 180)) {
            val1 = edgeImage[i + 1]; // i, j+1 right
          } else
          if ((r==0 && c==-1) && (0 <= angle && angle < 22.5 || 157.5 <= angle <= 180)) {
            val2 = edgeImage[i - 1]; // i, j-1 left
          } else
          // 45 angle
          if ((r==1 && c==-1) && (22.5 <= angle && angle < 67.5)) {
            val1 = edgeImage[i - 1 + WIDTH]; // i+1, j-1 bottom left
          } else
          if ((r==-1 && c==1) && (22.5 <= angle && angle < 67.5)) {
            val2 = edgeImage[i + 1 - WIDTH]; // i-1, j+1 top right
          } else
          // 90 angle
          if ((r==1 && c==0) && (67.5 <= angle && angle < 112.5)) {
            val1 = edgeImage[i + WIDTH]; // i+1, j bottom
          } else
          if ((r==-1 && c==0) && (67.5 <= angle && angle < 112.5)) {
            val2 = edgeImage[i - WIDTH]; // i-1, j top
          } else
          // 135 angle
          if ((r==-1 && c==-1) && (112.5 <= angle && angle < 157.5)) {
            val1 = edgeImage[i - 1 - WIDTH]; // i-1, j-1 top left
          } else
          if ((r==1 && c==1) && (112.5 <= angle && angle < 157.5)) {
            val2 = edgeImage[i + 1 + WIDTH]; // i+1, j+1 bottom right
          }
        }
      }
    } // end kernel
    if (edgeImage[i] >= val1 && edgeImage[i] >= val2) {
      suppressedImage[i] = edgeImage[i];
    } else {
      suppressedImage[i] = 0;
    }
  }
}

__global__ void gpuThreshold(unsigned char* suppressedImage, unsigned char* thresholdImage) {
  unsigned char weak = 25;
  unsigned char strong = 255;
  int offset, row, col, i, r, c;
  bool nextToStrong;
  // set weak and strong pixels
  for (i=0; i < WIDTH*HEIGHT; i++) {
    if (suppressedImage[i] > 50) suppressedImage[i] = strong;
    else if (suppressedImage[i] > 25) suppressedImage[i] = weak;
    else suppressedImage[i] = 0;
  }

  // hysteresis
  for (i=0; i<WIDTH*HEIGHT; i++) {
    row = i/WIDTH;
    col = i%WIDTH;
    nextToStrong = false;
    for (r=-1; r<2; r++) {
      for (c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=HEIGHT-1 && r > 0) || (col>=WIDTH-1 && c > 0)) {
          ;
        } else {
          if (suppressedImage[i]==weak) {
            offset = i+r*WIDTH+c;
            if (suppressedImage[offset] == strong) nextToStrong = true;
          }
        }
      }
    }
    if (nextToStrong) thresholdImage[i] = strong;
    else thresholdImage[i] = 0;
  }
}

__global__ void gpuCleanup(unsigned char* thresholdImage, unsigned char* finalImage) {
  int i;
  for (i=0; i< WIDTH*HEIGHT; i++) {
    if (thresholdImage[i] <255) finalImage[i] = 0;
    else finalImage[i] = 255;
  }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//***************************SERIAL VERSION*************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void serialCanny(unsigned char* originalImage, unsigned char* finalImage, int width, int height) {
  unsigned char *grayscaleImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *blurredImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *edgeImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  float *directions = (float*) malloc(width*height * sizeof(float));
  unsigned char *suppressedImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *thresholdImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));

  serialGrayscale(originalImage, grayscaleImage, width, height);
  serialGaussianBlur(grayscaleImage, blurredImage, width, height);
  serialEdgeDetection(blurredImage, edgeImage, directions, width, height);
  serialNonMaxSuppression(edgeImage, suppressedImage, directions, width, height);
  serialThreshold(suppressedImage, thresholdImage, width, height);
  serialCleanup(thresholdImage, finalImage, width, height);

}

void serialGrayscale(unsigned char* colorImage, unsigned char* grayImage, int width, int height) {
  for (int i=0; i < width*height; i++) {
    unsigned char temp = (colorImage[3*i] + colorImage[3*i+1] +  colorImage[3*i+2])/3;
    grayImage[i] = temp;
  }
}

void serialGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage, int width, int height) {
  unsigned char val;
  int offset, row, col;
  float gauss[3][3] = {{0.01, 0.08, 0.01}, {0.08, 0.64, 0.08}, {0.01, 0.08, 0.01}};

  for (int i=0; i<width*height; i++) {
    row = i/width;
    col = i%width;
    val = 0;
    for (int r=-1; r<2; r++) {
      for (int c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=height-1 && r > 0) || (col>=width-1 && c > 0)) {
          val += 0;
        } else {
          // printf("row: %d, col: %d, ", r, c);
          offset = i+r*width+c;
          val += grayImage[offset]*gauss[r+1][c+1];
        }
      }
    }
    blurredImage[i] = val;
  }
}

void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions, int width, int height) {
  double valx, valy, max, temp;
  int offset, row, col;
  int kx[3][3] = {{-1, 0, 1}, {-2, 0, -2}, {-1, 0, 1}};
  int ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  max = 0;
  for (int i=0; i<width*height; i++) {
    row = i/width;
    col = i%width;
    valx= 0;
    valy= 0;
    for (int r=-1; r<2; r++) {
      for (int c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=height-1 && r > 0) || (col>=width-1 && c > 0)) {
          valx+=0;
          valy+=0;
        } else {
          offset = i+r*width+c;
          valx += (double)blurredImage[offset]*kx[r+1][c+1];
          valy += (double)blurredImage[offset]*ky[r+1][c+1];
        }
      }
    }
    temp = sqrt(pow(valx, 2.0) + pow(valy, 2.0));
    if (temp > max) max = temp;
    edgeImage[i] = temp;
    directions[i] = atan(valy/valx);
  }
    for (int i=0; i<width*height; i++) {
      edgeImage[i] = edgeImage[i]/max*255;
    }
}

void serialNonMaxSuppression(unsigned char* edgeImage, unsigned char* suppressedImage, float* directions, int width, int height) {
  unsigned char val1, val2;
  int row, col;
  float angle;
  for (int i=0; i<width*height; i++) {
    val1 = 255;
    val2 = 255;
    row = i/width;
    col = i%width;
    angle = directions[i] * 180 / 3.141592653;
    if (angle < 0) angle += 180;

    for (int r=-1; r<2; r++) {
      for (int c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row==height-1 && r > 0) || (col==width-1 && c > 0)) {
          ;
        } else {
          // 0 angle
          if ((r==0 && c==1) && (0 <= angle && angle < 22.5 || 157.5 <= angle <= 180)) {
            val1 = edgeImage[i + 1]; // i, j+1 right
          } else
          if ((r==0 && c==-1) && (0 <= angle && angle < 22.5 || 157.5 <= angle <= 180)) {
            val2 = edgeImage[i - 1]; // i, j-1 left
          } else
          // 45 angle
          if ((r==1 && c==-1) && (22.5 <= angle && angle < 67.5)) {
            val1 = edgeImage[i - 1 + width]; // i+1, j-1 bottom left
          } else
          if ((r==-1 && c==1) && (22.5 <= angle && angle < 67.5)) {
            val2 = edgeImage[i + 1 - width]; // i-1, j+1 top right
          } else
          // 90 angle
          if ((r==1 && c==0) && (67.5 <= angle && angle < 112.5)) {
            val1 = edgeImage[i + width]; // i+1, j bottom
          } else
          if ((r==-1 && c==0) && (67.5 <= angle && angle < 112.5)) {
            val2 = edgeImage[i - width]; // i-1, j top
          } else
          // 135 angle
          if ((r==-1 && c==-1) && (112.5 <= angle && angle < 157.5)) {
            val1 = edgeImage[i - 1 - width]; // i-1, j-1 top left
          } else
          if ((r==1 && c==1) && (112.5 <= angle && angle < 157.5)) {
            val2 = edgeImage[i + 1 + width]; // i+1, j+1 bottom right
          }
        }
      }
    } // end kernel
    if (edgeImage[i] >= val1 && edgeImage[i] >= val2) {
      suppressedImage[i] = edgeImage[i];
    } else {
      suppressedImage[i] = 0;
    }
  }
}

void serialThreshold(unsigned char* suppressedImage, unsigned char* thresholdImage, int width, int height) {
  unsigned char weak = 25;
  unsigned char strong = 255;
  // set weak and strong pixels
  for (int i=0; i < width*height; i++) {
    if (suppressedImage[i] > 50) suppressedImage[i] = strong;
    else if (suppressedImage[i] > 25) suppressedImage[i] = weak;
    else suppressedImage[i] = 0;
  }

  // hysteresis
  int offset, row, col;
  bool nextToStrong;
  for (int i=0; i<width*height; i++) {
    row = i/width;
    col = i%width;
    nextToStrong = false;
    for (int r=-1; r<2; r++) {
      for (int c=-1; c<2; c++) {
        if ((row==0 && r < 0) || (col==0 && c < 0) || (row>=height-1 && r > 0) || (col>=width-1 && c > 0)) {
          ;
        } else {
          if (suppressedImage[i]==weak) {
            offset = i+r*width+c;
            if (suppressedImage[offset] == strong) nextToStrong = true;
          }
        }
      }
    }
    if (nextToStrong) thresholdImage[i] = strong;
    else thresholdImage[i] = 0;
  }
}

void serialCleanup(unsigned char* thresholdImage, unsigned char* finalImage, int width, int height) {
  for (int i=0; i< width*height; i++) {
    if (thresholdImage[i] <255) finalImage[i] = 0;
    else finalImage[i] = 255;
  }
}

void assertEqual(unsigned char* img1, unsigned char* img2, int n) {
  for (int i =0; i < n; i++) {
    // if( img1[i] != img2[i]) 
    // printf("%d: %u, %u\n", i, img1[i], img2[i]);
    assert(img1[i] == img2[i]);
  }
}
