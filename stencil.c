#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "omp.h"
#include <math.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image, int d0, int d1);
void init_image(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image);
void output_image(const char *file_name, const int nx, const int ny, const int width, const int height, float *image);
double wtime(void);

int main(int argc, char *argv[]){
  if (argc != 4){
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);
  int width = nx + 2;
  int height = ny + 2;
  int dim_0 = 0;
  int dim_1 = 0; 

  // Allocate the image
  float *image = _mm_malloc(sizeof(float) * width * height, 64);
  float *tmp_image = _mm_malloc(sizeof(float) * width * height, 64);
  init_image(nx, ny, width, height, image, tmp_image);

  int nthreads = omp_get_num_threads();
  double sq = sqrt(nthreads);
  int prelim_dim_0 = 0;
  int prelim_dim_1 = 0;

  for (int i = 1; i <= nthreads; i++){
    if (nthreads % i == 0){
      if (i > prelim_dim_0 && i <= (int)floor(sq)){
        prelim_dim_0 = i;
      }
    }
  }
  prelim_dim_1 = nthreads / prelim_dim_0;
  dim_0 = prelim_dim_0;
  dim_1 = prelim_dim_1;

  double tic = wtime();
  for (int t = 0; t < niters; ++t){
    stencil(nx, ny, width, height, image, tmp_image, dim_0, dim_1);
    stencil(nx, ny, width, height, tmp_image, image, dim_0, dim_1);
  }
  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, width, height, image);

}

void stencil(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image, int dim_0, int dim_1){

#pragma omp parallel shared(dim_0,dim_1,image,tmp_image,nx,ny,width,height) private(tid,coord_0,coord_1,tile_width,tile_height,y_start,y_end,x_start,x_end,h_halo_size,v_halo_size,pos)
{
  int tid = omp_get_thread_num();
  int coord_0 = (int)floor(tid / dim_0);
  int coord_1 = (int)(tid % dim_0);

     //calculate default tile nthreads (excl. halo)
    int tile_width = ceil((width - 2) / dim_0);   // Default tile nthreads, int col = dim_0;
    int tile_height = ceil((height - 2) / dim_1); // Default tile nthreads, int row = nthreads;
    //Tile receive bounds (excl. halo)
    int y_start = 1 + tile_height * (coord_1);
    int y_end = 0 + tile_height * (coord_1 + 1);
    int x_start = 1 + tile_width * (coord_0);
    int x_end = 0 + tile_width * (coord_0 + 1);
    //Halo information, encodes actual tile nthreads
    int h_halo_size = tile_width + 2;  //For complete tile
    int v_halo_size = tile_height + 2; //For complete tile
    if (coord_0 == dim_0 - 1){
      x_end = nx;
      tile_width = x_end - x_start + 1;
      h_halo_size = tile_width + 2;
    }
    if (coord_1 == dim_1 - 1){
      y_end = ny;
      tile_height = y_end - y_start + 1;
      v_halo_size = tile_height + 2;
    }

    int pos = y_start * width + x_start;
    
#pragma omp parallel for collapse(2)
    for (int iy = y_start; iy <= y_end; iy++){
      for (int ix = x_start; ix <= x_start; ix++){
        tmp_image[pos] = image[pos] * 0.6f + (image[pos - 1] + image[pos + 1] + image[pos - width] + image[pos + width]) * 0.1f;
        pos += 1;
      }
      pos += 2;
    }

}
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image){
  // Zero everything
  for (int i = 0; i < nx + 2; ++i){
    for (int j = 0; j < ny + 2; ++j){
      image[i + j * width] = 0.0;
      tmp_image[i + j * width] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int ib = 0; ib < nx; ib += tile_size){
    for (int jb = 0; jb < ny; jb += tile_size){
      if ((ib + jb) % (tile_size * 2)){
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int i = ib + 1; i < ilim + 1; ++i){
          for (int j = jb + 1; j < jlim + 1; ++j){
            image[i + j * width] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char *file_name, const int nx, const int ny, const int width, const int height, float *image){
  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp){
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int i = 1; i < nx + 1; ++i){
    for (int j = 1; j < ny + 1; ++j){
      if (image[i + j * width] > maximum)
        maximum = image[i + j * width];
    }
  }

  // Output image, converting to numbers 0-255
  for (int i = 1; i < nx + 1; ++i){
    for (int j = 1; j < ny + 1; ++j){
      fputc((char)(255.0 * image[i + j * width] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
