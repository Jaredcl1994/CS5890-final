#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

void serialGrayscale(unsigned char* colorImage, unsigned char* grayImage, const int width, const int height);
void serialGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage, const int width, const int height);
void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions, const int width, const int height);
void serialNonMaxSuppression(unsigned char* edgeImage, unsigned char* suppressedImage, float* directions, int width, int height);
void serialThreshold(unsigned char* suppressedImage, unsigned char* thresholdImage, int width, int height);
void serialCleanup(unsigned char* thresholdImage, unsigned char* finalImage, int width, int height);

int main() {
  const int width = 3379;
  const int height = 3005;
  const int n = width*height*3;
  unsigned char *originalImage = (unsigned char*) malloc(n * sizeof(unsigned char));
  unsigned char *grayscaleImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *blurredImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *edgeImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  float *directions = (float*) malloc(width*height * sizeof(float));
  unsigned char *suppressedImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *thresholdImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *finalImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));

  // read image into originalImage
  FILE *imageFile;
  imageFile = fopen("butterfly_nebula_3379x3005_rgb.raw", "rb");
  fread(originalImage, sizeof(unsigned char), n, imageFile);
  fclose(imageFile);

  // serial canny edge detection
  serialGrayscale(originalImage, grayscaleImage, width, height);
  serialGaussianBlur(grayscaleImage, blurredImage, width, height);
  serialEdgeDetection(blurredImage, edgeImage, directions, width, height);
  serialNonMaxSuppression(edgeImage, suppressedImage, directions, width, height);
  serialThreshold(suppressedImage, thresholdImage, width, height);
  serialCleanup(thresholdImage, finalImage, width, height);

  // openmp canny edge detection
  // distributed canny edge detection with ghost/halo regions exchange
  // distributed canny edge detection without ghost/halo regions exchange (ghost regions preloaded)
  // gpu canny edge detection (different file)

  FILE *outputFile;
  // outputFile = fopen("grayscaleImage.raw", "wb+");
  // fwrite(grayscaleImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);

  // outputFile = fopen("blurredImage.raw", "wb+");
  // fwrite(blurredImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);

  // outputFile = fopen("edgeImage.raw", "wb+");
  // fwrite(edgeImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);

  // outputFile = fopen("suppressedImage.raw", "wb+");
  // fwrite(suppressedImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);

  // outputFile = fopen("thresholdImage.raw", "wb+");
  // fwrite(thresholdImage, sizeof(unsigned char), width*height, outputFile);
  // fclose(outputFile);

  outputFile = fopen("finalImage.raw", "wb+");
  fwrite(finalImage, sizeof(unsigned char), width*height, outputFile);
  fclose(outputFile);
}

void serialGrayscale(unsigned char* colorImage, unsigned char* grayImage, const int width, const int height) {
  for (int i=0; i < width*height; i++) {
    unsigned char temp = (colorImage[3*i] + colorImage[3*i+1] +  colorImage[3*i+2])/3;
    grayImage[i] = temp;
  }
}

void serialGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage, const int width, const int height) {
  unsigned char val;
  int offset, row, col;
  float gauss[3][3] = {{0.01, 0.08, 0.01}, {0.08, 0.64, 0.08}, {0.01, 0.08, 0.01}};

  for (int i=0; i<width*height; i++) {
    int row = i/width;
    int col = i%width;
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

void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, float* directions, const int width, const int height) {
  double valx, valy, max, temp;
  int offset, row, col;
  int kx[3][3] = {{-1, 0, 1}, {-2, 0, -2}, {-1, 0, 1}};
  int ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  max = 0;
  for (int i=0; i<width*height; i++) {
    int row = i/width;
    int col = i%width;
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
  for (int i=0; i<width*height; i++) {
    int row = i/width;
    int col = i%width;
    bool nextToStrong = false;
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