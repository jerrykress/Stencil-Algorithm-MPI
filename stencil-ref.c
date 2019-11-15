
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define OUTPUT_FILE "stencil.pgm"
#define NDIMS 2
#define MASTER 0

void stencil(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
void halo_exchange(MPI_Comm comm_cart, float * tile, int v_halo_size, int h_halo_size, int rows, int cols, int * coords, int * dims);
void get_dimensions(int rank, int * dims, int * coords, int size, int nx, int ny, int *h_halo_size, int *v_halo_size, int *cols, int *rows, int *startCoods);
double wtime(void);

int main(int argc, char *argv[]) {
    int rank;
    int size;
    int flag;
    int tag = 0;
    int reorder = 0;
    int dims[NDIMS];
    int periods[NDIMS];
    int coords[NDIMS];
    MPI_Comm comm_cart;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Initialized(&flag);
    if ( flag != 1 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (argc != 4) {
        fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int niters = atoi(argv[3]);

    if (rank == MASTER) {
        printf("nx %d ny %d niters %d\n", nx, ny, niters);
    }

    float *image = (float *)_mm_malloc(sizeof(float)*nx*ny,64);
    float *tmp_image = (float *)_mm_malloc(sizeof(float)*nx*ny,64);
    init_image(nx, ny, image, tmp_image);

    double maxTime = 0;

   /*
    ** If only one core, run serially
    */
    if (size == 1) {
        // Call the stencil kernel
        double tic = wtime();
        for (int t = 0; t < niters; ++t) {
            stencil(nx, ny, image, tmp_image);
            stencil(nx, ny, tmp_image, image);
        }
        double toc = wtime();

        // Output
        printf("------------------------------------\n");
        printf(" process %d of %d runtime: %lf s\n", rank, size, toc-tic);
        printf("------------------------------------\n");

        output_image(OUTPUT_FILE, nx, ny, image);
    } else {
       /*
        ** Setup cartesian grid
        */

       //dims - integer array of size ndims specifying the number of nodes in each dimension. A value of 0 indicates that MPI_Dims_create should fill in a suitable value. 
        for (int i=0; i<NDIMS; i++) {
         dims[i] = 0;
         periods[i] = 0;
        }

        MPI_Dims_create(size, 2, dims);
        MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &comm_cart);
        MPI_Cart_coords(comm_cart, rank, NDIMS, coords);
        MPI_Barrier(MPI_COMM_WORLD);

        int h_halo_size = 0;
        int v_halo_size = 0;
        int cols = 0;
        int rows = 0;
        int startCoords[2] = {0, 0};
        get_dimensions(rank, dims, coords, size, nx, ny, &h_halo_size, &v_halo_size, &cols, &rows, startCoords);

        // Get your tile
        float *tile = (float *)_mm_malloc(sizeof(float)*cols*rows,64);
        float *tmp_tile = (float *)_mm_malloc(sizeof(float)*cols*rows,64);
        for (int i = 0; i < cols; i++) {
           for (int j = 0; j < rows; j++) {
               tile[j+i*rows] = image[(j+startCoords[1])+(i+startCoords[0])*ny];
           }
        }

        // Call the stencil kernel
        double tic = wtime();
        for (int t = 0; t < niters; ++t) {
           halo_exchange(comm_cart, tile, v_halo_size, h_halo_size, rows, cols, coords, dims);
           stencil(cols, rows, tile, tmp_tile);
           halo_exchange(comm_cart, tmp_tile, v_halo_size, h_halo_size, rows, cols, coords, dims);
           stencil(cols, rows, tmp_tile, tile);
        }
        double toc = wtime();

        if (rank == MASTER) {
           // Save rank 0 tile
           for (int i = 0; i < cols; i++) {
               for (int j = 0; j < rows; j++) {
                   image[(j+startCoords[1])+(i+startCoords[0])*ny] = tile[j+i*rows];
               }
           }

           // Receive tiles from cores
           int r_info[4] = {0, 0, 0, 0};
           for (int r = 1; r < size; r++) {
               MPI_Recv(&r_info[0], BUFSIZ, MPI_INT, r, tag, MPI_COMM_WORLD, &status);
               for (int i = 0; i < r_info[3]; i++) {
                   MPI_Recv(&image[(r_info[1])+(i+r_info[0])*(ny)], BUFSIZ, MPI_FLOAT, r, tag, MPI_COMM_WORLD, &status);
               }
           }

           maxTime = toc-tic;
           double rTime = 0;
           for (int r = 1; r < size; r++) {
               MPI_Recv(&rTime, BUFSIZ, MPI_DOUBLE, r, tag, MPI_COMM_WORLD, &status);
               if (rTime > maxTime) maxTime = rTime;
           }

            // Output
            printf("------------------------------------\n");
            printf(" process %d of %d runtime: %lf s\n", rank, size, maxTime);
            printf("------------------------------------\n");

            output_image(OUTPUT_FILE, nx, ny, image);
        } else {
            // Tell master where your tile should start
            int x = 0;
            int y = 0;
            int m_cols = cols;
            int m_rows = rows;
            if (coords[0] != 0) {
                startCoords[0] += 1;
                x += 1;
            }
            if (coords[1] != dims[1]-1) {
                startCoords[1] += 1;
                y += 1;
            }
            if ((coords[0] == dims[0]-1) || (coords[0] == 0)) {
                m_cols -= 1;
            } else {
                m_cols -= 2;
            }
            if ((coords[1] == dims[1]-1) || (coords[1] == 0)) {
                m_rows -= 1;
            } else {
                m_rows -= 2;
            }
            int m[4] = {startCoords[0], startCoords[1], m_rows, m_cols};
            MPI_Send(&m[0], 4, MPI_INT, 0, tag, MPI_COMM_WORLD);
            for (int i = 0; i < m_cols; i++) {
                MPI_Send(&tile[(y)+(i+x)*(rows)], m_rows, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
            }

            double time = toc-tic;
            MPI_Send(&time, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }

        _mm_free(tile);
        _mm_free(tmp_tile);
    }

    _mm_free(image);
    _mm_free(tmp_image);

    MPI_Finalize(); /* finialise the MPI enviroment */
}

void get_dimensions(int rank, int * dims, int * coords, int size, int nx, int ny,
                    int *h_halo_size, int *v_halo_size, int *cols, int *rows, int *startCoords) {
    /*
     ** Figure out halo sizes
     */
    if (coords[0] == dims[0]-1) *h_halo_size = nx-(nx/dims[0])*(dims[0]-1);
    else *h_halo_size = nx/dims[0];
    if (coords[1] == dims[1]-1) *v_halo_size = ny-(ny/dims[1])*(dims[1]-1);
    else *v_halo_size = ny/dims[1];

    /*
     ** Figure out tile sizes
     */
    if ((coords[0] == 0) || (coords[0] == dims[0]-1)) *cols = *h_halo_size + 1;
    else *cols = *h_halo_size + 2;
    if ((coords[1] == 0) || (coords[1] == dims[1]-1)) *rows = *v_halo_size + 1;
    else *rows = *v_halo_size + 2;

   /*
    ** Figure out position on total image
    */
    startCoords[0] = coords[0];
    startCoords[1] = dims[1]-coords[1]-1;
    int extraRows = 0;
    int rowsPerP = ny/dims[1];
    if (ny%size != 0) {
        extraRows = ny-(ny/dims[1])*dims[1];
    }
    if (startCoords[1] != 0) {
        startCoords[1] = extraRows + (startCoords[1])*rowsPerP - 1;
    }
    int colsPerP = nx/dims[0];
    startCoords[0] = (startCoords[0])*colsPerP;
    if (startCoords[0] != 0) startCoords[0] -= 1;
}

void halo_exchange(MPI_Comm comm_cart, float * tile, int v_halo_size, int h_halo_size, int rows, int cols, int * coords, int * dims) {
    int counts[4] = {v_halo_size, v_halo_size, h_halo_size, h_halo_size};
    int displs[4] = {0, h_halo_size, 2*h_halo_size, 2*h_halo_size+v_halo_size};

    float *sendbuf = (float *)_mm_malloc(sizeof(float)*(2*h_halo_size+2*v_halo_size), 2*h_halo_size+2*v_halo_size);
    float *recvbuf = (float *)_mm_malloc(sizeof(float)*(2*h_halo_size+2*v_halo_size), 2*h_halo_size+2*v_halo_size);
    int vdisp = 0;
    int hdisp = 0;

    if (coords[1] == dims[1]-1) vdisp = 0;
    else vdisp = 1;
    if (coords[0] == 0) hdisp = 0;
    else hdisp = 1;

   /*
    ** Setup the send buffer.
    */
    // West
    for (int j = 0; j < v_halo_size; j++) {
        sendbuf[displs[0]+j] = tile[vdisp+j+(rows)];
    }
    // East
    for (int j = 0; j < v_halo_size; j++) {
        sendbuf[displs[1]+j] = tile[(vdisp+j)+(cols-1)*(rows)-rows];
    }
    // South
    for (int i = 0; i < h_halo_size; i++) {
        sendbuf[displs[2]+i] = tile[(rows-1)+(i+hdisp)*(rows)-1];
    }
    // North
    for (int i = 0; i < h_halo_size; i++) {
        sendbuf[displs[3]+i] = tile[(i+hdisp)*(rows)+1];
    }

   /*
    ** Exchange.
    */
    MPI_Neighbor_alltoallv(sendbuf, counts, displs, MPI_FLOAT, recvbuf, counts, displs, MPI_FLOAT, comm_cart);

   /*
    ** Save halos.
    */
    // West
    if (coords[0] != 0) {
        for (int j = 0; j < v_halo_size; j++) {
            tile[vdisp+j] = recvbuf[displs[0]+j];
        }
    }
    // East
    if (coords[0] != dims[0]-1) {
        for (int j = 0; j < v_halo_size; j++) {
            tile[(vdisp+j)+(cols-1)*(rows)] = recvbuf[displs[1]+j];
        }
    }
    // South
    if (coords[1] != 0) {
        for (int i = 0; i < h_halo_size; i++) {
            tile[(rows-1)+(i+hdisp)*(rows)] = recvbuf[displs[2]+i];
        }
    }
    // North
    if (coords[1] != dims[1]-1) {
        for (int i = 0; i < h_halo_size; i++) {
            tile[(i+hdisp)*(rows)] = recvbuf[displs[3]+i];
        }
    }

    _mm_free(sendbuf);
    _mm_free(recvbuf);
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
    // MIDDLE
    for (int i = 1; i < nx-1; ++i) {
        __assume_aligned(image, 64);
        __assume_aligned(tmp_image, 64);
        for (int j = 1; j < ny-1; ++j) {
            tmp_image[j+i*ny] = image[j+i*ny] * 0.6f;
            tmp_image[j+i*ny] += image[j+i*ny-ny] * 0.1f;
            tmp_image[j+i*ny] += image[j+i*ny+ny] * 0.1f;
            tmp_image[j+i*ny] += image[j+i*ny-1] * 0.1f;
            tmp_image[j+i*ny] += image[j+i*ny+1] * 0.1f;
        }
    }

    // LEFT COLUMN
    __assume_aligned(image, 64);
    __assume_aligned(tmp_image, 64);
    for (int i = 1; i < ny-1; ++i) {
        tmp_image[i] = image[i] * 0.6f;
        tmp_image[i] += image[i+ny] * 0.1f;
        tmp_image[i] += image[i-1] * 0.1f;
        tmp_image[i] += image[i+1] * 0.1f;
    }

    // TOP ROW
    for (int j = 1; j < nx-1; j++) {
        tmp_image[j*ny] = image[j*ny] * 0.6f;
        tmp_image[j*ny] += image[j*ny-ny] * 0.1f;
        tmp_image[j*ny] += image[j*ny+ny] * 0.1f;
        tmp_image[j*ny] += image[j*ny+1] * 0.1f;
    }

    // BOTTOM ROW
    for (int j = 1; j < nx-1; j++) {
        tmp_image[(ny-1)+j*ny] = image[(ny-1)+j*ny] * 0.6f;
        tmp_image[(ny-1)+j*ny] += image[(ny-1)+j*ny-ny] * 0.1f;
        tmp_image[(ny-1)+j*ny] += image[(ny-1)+j*ny+ny] * 0.1f;
        tmp_image[(ny-1)+j*ny] += image[(ny-1)+j*ny-1] * 0.1f;
    }

    // RIGHT COLUMN
    __assume_aligned(image, 64);
    __assume_aligned(tmp_image, 64);
    for (int i = 1; i < ny-1; ++i) {
        tmp_image[(nx-1)*ny+i] = image[(nx-1)*ny+i] * 0.6f;
        tmp_image[(nx-1)*ny+i] += image[(nx-1)*ny+i-ny] * 0.1f;
        tmp_image[(nx-1)*ny+i] += image[(nx-1)*ny+i-1] * 0.1f;
        tmp_image[(nx-1)*ny+i] += image[(nx-1)*ny+i+1] * 0.1f;
    }

    // TOP LEFT CORNER
    tmp_image[0] = image[0] * 0.6f;
    tmp_image[0] += image[ny] * 0.1f;
    tmp_image[0] += image[1] * 0.1f;

    // BOTTOM LEFT CORNER
    tmp_image[ny-1] = image[ny-1] * 0.6f;
    tmp_image[ny-1] += image[ny-2] * 0.1f;
    tmp_image[ny-1] += image[2*ny-1] * 0.1f;

    // TOP RIGHT CORNER
    tmp_image[ny*(nx-1)] = image[ny*(nx-1)] * 0.6f;
    tmp_image[ny*(nx-1)] += image[ny*(nx-1)+1] * 0.1f;
    tmp_image[ny*(nx-1)] += image[ny*(nx-1)-ny] * 0.1f;

    // BOTTOM RIGHT CORNER
    tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f;
    tmp_image[nx*ny-1] += image[nx*ny-2] * 0.1f;
    tmp_image[nx*ny-1] += image[nx*ny-1-ny] * 0.1f;
}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
