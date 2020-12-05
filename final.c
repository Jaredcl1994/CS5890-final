#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

void serialGrayscale(unsigned char* colorImage, unsigned char* grayImage, const int width, const int height);
void serialGaussianBlur(unsigned char* grayImage, unsigned char* blurredImage, const int width, const int height);
// void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, const int width, const int height);

int main() {
  const int width = 3379;
  const int height = 3005;
  const int n = width*height*3;
  unsigned char *originalImage = (unsigned char*) malloc(n * sizeof(unsigned char));
  unsigned char *grayscaleImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *blurredImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));
  unsigned char *edgeImage = (unsigned char*) malloc(width*height * sizeof(unsigned char));

  // read image into originalImage
  FILE *imageFile;
  imageFile = fopen("butterfly_nebula_3379x3005_rgb.raw", "rb");
  fread(originalImage, sizeof(unsigned char), n, imageFile);
  fclose(imageFile);

  serialGrayscale(originalImage, grayscaleImage, width, height);
  serialGaussianBlur(grayscaleImage, blurredImage, width, height);
  // serialEdgeDetection(blurredImage, edgeImage, width, height);

  FILE *outputFile;
  outputFile = fopen("grayscaleImage.raw", "wb+");
  fwrite(grayscaleImage, sizeof(unsigned char), width*height, outputFile);
  fclose(outputFile);

  outputFile = fopen("blurredImage.raw", "wb+");
  fwrite(blurredImage, sizeof(unsigned char), width*height, outputFile);
  fclose(outputFile);

  outputFile = fopen("edgeImage.raw", "wb+");
  fwrite(edgeImage, sizeof(unsigned char), width*height, outputFile);
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

// void serialEdgeDetection(unsigned char* blurredImage, unsigned char* edgeImage, const int width, const int height) {
  // unsigned char valx, valy;
  // double temp;
  // int offset, row, col;
  // int kx[3][3] = {{-1, 0, 1}, {-2, 0, -2}, {-1, 0, 1}};
  // int ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
//
  // for (int i=0; i<width*height; i++) {
    // int row = i/width;
    // int col = i%width;
    // valx= 0;
    // valy= 0;
    // for (int r=-1; r<2; r++) {
      // for (int c=-1; c<2; c++) {
        // if (!(row==0 && r < 0) || (col==0 && c < 0) || (row>=height-1 && r > 0) || (col>=width-1 && c > 0)) {
          // printf("row: %d, col: %d, ", r, c);
          // offset = i+r*width+c;
          // valx += blurredImage[offset]*kx[r+1][c+1];
          // valy += blurredImage[offset]*ky[r+1][c+1];
        // }
      // }
    // }
    // temp = valx*valx + valy*valy;
    // edgeImage[i] = (unsigned char) sqrt(temp);
  // }
//
//
// }
