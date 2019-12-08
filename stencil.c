#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include "omp.h"
#include <math.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image);
void init_image(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image);
void output_image(const char *file_name, const int nx, const int ny, const int width, const int height, float *image);
void exchange(MPI_Comm comcord, float *tile, int h_halo_size, int v_halo_size, int *coords, int *dims, MPI_Request *request);
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

  // Allocate the image
  float *image = _mm_malloc(sizeof(float) * width * height, 64);
  float *tmp_image = _mm_malloc(sizeof(float) * width * height, 64);
  init_image(nx, ny, width, height, image, tmp_image);

  #pragma omp parallel shared(nthreads) private(tid, tile, tmp_tile)
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    clock_t tic, toc;
    float elapsed_time;
    double maxr;

    #pragma omp master
    if (nthreads == 1){
      // Call the stencil kernel
      double tic = clock();
      for (int t = 0; t < niters; ++t){
        stencil(nx, ny, width, height, image, tmp_image);
        stencil(nx, ny, width, height, tmp_image, image);
      }
      double toc = clock();

      elapsed_time = (float)(toc - tic) / (float)CLOCKS_PER_SEC;
      elapsed_time /= (float)nthreads;

      // Output
      printf("------------------------------------\n");
      printf(" runtime: %f s\n", elapsed_time);
      printf("------------------------------------\n");

      output_image(OUTPUT_FILE, nx, ny, width, height, image);
    }
    else

    {
      //calculate default tile size (excl. halo)
      int tile_width = nx; 
      int tile_height = ceil((height - 2) / nthreads); 
      //Tile receive bounds (excl. halo)
      int y_start = 1 + tile_height * (tid);
      int y_end = 0 + tile_height * (tid + 1);
      int x_start = 1;
      int x_end = nx;
      //Halo information, encodes actual tile size
      int h_halo_size = tile_width + 2; 
      int v_halo_size = tile_height + 2; 
      if (tid == nthreads - 1){
        y_end = ny;
        tile_height = y_end - y_start + 1;
        v_halo_size = tile_height + 2;
      }
      //Setup this tile
      float *tile = _mm_malloc(sizeof(float) * h_halo_size * v_halo_size, 64);
      float *tmp_tile = _mm_malloc(sizeof(float) * h_halo_size * v_halo_size, 64);

      //Copy from original image (excl. halo)
      for (int y_offset = 0; y_offset < tile_height; y_offset++){
        for (int x_offset = 0; x_offset < tile_width; x_offset++){
          tile[(y_offset + 1) * h_halo_size + (x_offset + 1)] = image[(y_start + y_offset) * width + (x_start + x_offset)];
          tmp_tile[(y_offset + 1) * h_halo_size + (x_offset + 1)] = image[(y_start + y_offset) * width + (x_start + x_offset)];
        }
      }

      // Call the stencil kernel for iters
      for (int t = 0; t < niters; ++t){
        stencil(tile_width, tile_height, h_halo_size, v_halo_size, tile, tmp_tile);
        stencil(tile_width, tile_height, h_halo_size, v_halo_size, tmp_tile, tile);
      }
      double toc = wtime();

      if (rank == MASTER){
        //save rank 0 to original image (excl. halo)
        for (int y_offset = 0; y_offset < tile_height; y_offset++){
          for (int x_offset = 0; x_offset < tile_width; x_offset++){
            image[(y_start + y_offset) * width + (x_start + x_offset)] = tile[(y_offset + 1) * h_halo_size + (x_offset + 1)];
          }
        }

        //save other tiles
        for (int receive_rank = 1; receive_rank < size; receive_rank++){
          int tile_meta_i[4] = {0, 0, 0, 0}; //x_start, y_start, x_end, y_end
          MPI_Recv(&tile_meta_i[0], BUFSIZ, MPI_INT, receive_rank, tag, MPI_COMM_WORLD, &status);
          printf("Merging from rank: %d, xs: %d, ys: %d, xe: %d, ye: %d\n", receive_rank, tile_meta_i[0], tile_meta_i[1], tile_meta_i[2], tile_meta_i[3]);
          for (int i = tile_meta_i[1]; i < tile_meta_i[3] + 1; i++){
            MPI_Recv(&image[i * width + tile_meta_i[0]], BUFSIZ, MPI_FLOAT, receive_rank, tag, MPI_COMM_WORLD, &status);
          }
        }

        printf("------------------------------------\n");
        printf(" runtime: %lf s\n", toc - tic);
        printf("------------------------------------\n");

        output_image(OUTPUT_FILE, nx, ny, width, height, image);
      }
      else
      {
        //send to rank 0
        int tile_meta_o[4] = {x_start, y_start, x_end, y_end};
        MPI_Send(&tile_meta_o[0], 4, MPI_INT, 0, tag, MPI_COMM_WORLD);
        for (int i = 1; i < v_halo_size - 1; i++){
          MPI_Send(&tile[i * h_halo_size + 1], tile_width, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
        }
      }

      _mm_free(tile);
      _mm_free(tmp_tile);
    }
  
  }

  _mm_free(image);
  _mm_free(tmp_image);

  MPI_Finalize();
}

void exchange(MPI_Comm comcord, float *tile, int h_halo_size, int v_halo_size, int *coords, int *dims, MPI_Request *request){
  int counts[4] = {v_halo_size, v_halo_size, h_halo_size, h_halo_size};
  int offsets[4] = {0, v_halo_size, 2 * v_halo_size, 2 * v_halo_size + h_halo_size}; 
  float *s_buffer = _mm_malloc(sizeof(float) * (2 * h_halo_size + 2 * v_halo_size), 2 * h_halo_size + 2 * v_halo_size);
  float *r_buffer = _mm_malloc(sizeof(float) * (2 * h_halo_size + 2 * v_halo_size), 2 * h_halo_size + 2 * v_halo_size);

  // LEFT
  for (int i = 0; i < counts[0] - 1; i++){
    s_buffer[i + offsets[0]] = tile[i * h_halo_size + 1]; //(1,i), i~[0,v_halo)
  }
  // RIGHT
  for (int i = 0; i < counts[1] - 1; i++){
    s_buffer[i + offsets[1]] = tile[(i + 1) * h_halo_size - 2]; //(h_halo_size -2,i)
  }
  // UP
  for (int i = 0; i < counts[2] - 1; i++){
    s_buffer[i + offsets[2]] = tile[1 * h_halo_size + i]; //(i,1)
  }
  // DOWN
  for (int i = 0; i < counts[3] - 1; i++){
    s_buffer[i + offsets[3]] = tile[(v_halo_size - 2) * h_halo_size + i]; //(i,v_halo_size -2)
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Ineighbor_alltoallv(s_buffer, counts, offsets, MPI_FLOAT, r_buffer, counts, offsets, MPI_FLOAT, comcord, request);
  MPI_Barrier(MPI_COMM_WORLD);

  // LEFT
  for (int i = 0; i < counts[0] - 1; i++){
    tile[i * h_halo_size] = r_buffer[i + offsets[0]]; //(0,i)
  }
  // RIGHT
  for (int i = 0; i < counts[1] - 1; i++){
    tile[(i + 1) * h_halo_size - 1] = r_buffer[i + offsets[1]]; //(h_halo_size -1,i)
  }
  // UP
  for (int i = 0; i < counts[2] - 1; i++){
    tile[i] = r_buffer[i + offsets[2]]; //(i,0)
  }
  // DOWN
  for (int i = 0; i < counts[3] - 1; i++){
    tile[(v_halo_size - 1) * h_halo_size + i] = r_buffer[i + offsets[3]]; //(i,v_halo_size -1)
  }

  _mm_free(s_buffer);
  _mm_free(r_buffer);
}

void stencil(const int nx, const int ny, const int width, const int height, float *image, float *tmp_image){
  int delta = width - nx;
  int pos = 1 + width;
  for (int iy = 0; iy < ny; iy++){
    __assume_aligned(image, 64);
    __assume_aligned(tmp_image, 64);
    for (int ix = 0; ix < nx; ix++){
      tmp_image[pos] = image[pos] * 0.6f + (image[pos - 1] + image[pos + 1] + image[pos - width] + image[pos + width]) * 0.1f;
      pos += 1;
    }
    pos += delta;
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
