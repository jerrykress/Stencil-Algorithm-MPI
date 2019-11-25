#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <math.h> 

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
void exchange(MPI_Comm comcord, float * tile, int h_halo_size, int v_halo_size, int * coords, int * dims);
double wtime(void);

int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int rank;
  int size;
  int dest;              /* destination rank for message */
  int source;            /* source rank of a message */
  int tag = 0;           /* scope for adding extra information to a message */
  int dims[2] = {0,0};   //Leave 0 as default for MPI automatic config
  int periods[2] = {0,0};
  int coords[2] = {0,0};
  int reorder = 0;       //TODO:find out if allow reorder makes it run faster
  MPI_Comm comcord;
  MPI_Status status;     /* struct used by MPI_Recv */
  char message[BUFSIZ];

  /* MPI_Init returns once it has started up processes */
  MPI_Init( &argc, &argv );

  /* size and rank will become ubiquitous */ 
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  //=====================================================================

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  
  if(size == 1){
    // Call the stencil kernel
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      stencil(nx, ny, width, height, image, tmp_image);
      stencil(nx, ny, width, height, tmp_image, image);
    }
    double toc = wtime();

    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, width, height, image);

  } else {

    //run on multiple processes, set up a grid
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comcord);
    MPI_Cart_coords(comcord, rank, 2, coords); //Get cartesian co-ordinates of a particular rank:
    MPI_Barrier(MPI_COMM_WORLD);
    printf("this is tile coord[0]: %d, coord[1]: %d \n", coords[0], coords[1]);

    //calculate default tile size (excl. halo)
    int tile_width = ceil((width - 2)/dims[0]); // Default tile size, int col = dims[0];
    int tile_height = ceil((height -2)/dims[1]); // Default tile size, int row = dims[1];
    //Tile receive bounds (excl. halo)
    int y_start = 1 + tile_height * (coords[1]);
    int y_end   = 0 + tile_height * (coords[1] + 1);
    if(y_end > ny) y_end = ny; //Adjust for incomplete tile
    int x_start = 1 + tile_width  * (coords[0]);
    int x_end   = 0 + tile_width  * (coords[0] + 1);
    if(x_end > nx) x_end = nx; //Adjust for incomplete tile
    //Halo information, encodes actual tile size
    int h_halo_size = tile_width + 2; //For complete tile
    int v_halo_size = tile_height + 2;  //For complete tile
    if (coords[0] == dims[0] - 1){
      h_halo_size = nx - (dims[0] - 1) * tile_width + 2; //Adjust for incomplete tile
      tile_width = h_halo_size - 2;
    } 
    if (coords[1] == dims[1] - 1){
      v_halo_size = ny - (dims[1] - 1) * tile_height + 2; //Adjust for incomplete tile
      tile_height = v_halo_size - 2;
    }
    //Setup this tile
    float* tile = malloc(sizeof(float) * (tile_width + 2) * (tile_height + 2));
    float* tmp_tile = malloc(sizeof(float) * (tile_width + 2) * (tile_height + 2));
    printf("Finished tile setup on process: %d\n", rank);
    printf("Process %d info: %d %d %d %d %d %d %d %d\n", rank, x_start, x_end, y_start, y_end, tile_width, tile_width, h_halo_size, v_halo_size);

    //Copy from original image (incl. halo)
    for(int y_offset = 0; y_offset < tile_height + 2; y_offset++){
      for(int x_offset = 0 ; x_offset < tile_width + 2; x_offset++){
        tile[(y_offset + 1) * h_halo_size + (x_offset + 1)]     = image[(y_start + y_offset) * width + (x_start + x_offset)];
        tmp_tile[(y_offset + 1) * h_halo_size + (x_offset + 1)] = image[(y_start + y_offset) * width + (x_start + x_offset)];
      }
    }
    printf("Finished tile copy on process: %d\n", rank);

    // Call the stencil kernel for iters
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      printf("Starting stencil on process: %d\n", rank);
      exchange(comcord, tile, h_halo_size, v_halo_size, coords, dims);
      stencil(tile_width, tile_height, h_halo_size, v_halo_size, tile, tmp_tile);
      exchange(comcord, tmp_tile, h_halo_size, v_halo_size, coords, dims);
      stencil(tile_width, tile_height, h_halo_size, v_halo_size, tmp_tile, tile);
      //exchange(comcord, tile, h_halo_size, v_halo_size, coords, dims);
    }
    double toc = wtime();

    if(rank == MASTER){
      //save rank 0 to original image (excl. halo)
      for (int y_offset = 0; y_offset < tile_height; y_offset++){
        for (int x_offset = 0; x_offset < tile_width; x_offset++){
          image[(y_start + y_offset) * width + (x_start + x_offset)]  = tile[(y_offset + 1) * h_halo_size + (x_offset + 1)];
        }
      }

      //save other tiles
      for (int receive_rank = 1; receive_rank < size; receive_rank ++){
        int tile_meta_i[4] = {0, 0, 0, 0}; //x_start, y_start, tile_width, y_end
        MPI_Recv(&tile_meta_i[0], BUFSIZ, MPI_INT, receive_rank, tag, MPI_COMM_WORLD, &status);
        for(int i = tile_meta_i[1]; i < tile_meta_i[3] + 1; i++){
          MPI_Recv(&image[i * width + tile_meta_i[0]], BUFSIZ, MPI_FLOAT, receive_rank, tag, MPI_COMM_WORLD, &status);
        }
      }

      printf("------------------------------------\n");
      printf(" runtime: %lf s\n", toc - tic);
      printf("------------------------------------\n");

      output_image(OUTPUT_FILE, nx, ny, width, height, image);

    } else {
      //send to rank 0
      int tile_meta_o[4] = {x_start, y_start, tile_width, y_end};
      MPI_Send(&tile_meta_o[0], 4, MPI_INT, 0, tag, MPI_COMM_WORLD);
      for(int i = 1; i < v_halo_size - 1; i++){
        MPI_Send(&tile[i*h_halo_size + 1], tile_width, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
      }

    }

    free(tile);
    free(tmp_tile);
  }

  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void exchange(MPI_Comm comcord, float * tile, int h_halo_size, int v_halo_size, int * coords, int * dims){
    //float * tile, int rows, int cols
    //Halo order: UP, DOWN, LEFT, RIGHT
    int counts[4] = {v_halo_size, v_halo_size, h_halo_size, h_halo_size};
    int offsets[4] = {0, v_halo_size, 2 * v_halo_size, 2 * v_halo_size + h_halo_size};

    float *s_buffer = malloc(sizeof(float) * (2*h_halo_size + 2*v_halo_size));
    float *r_buffer = malloc(sizeof(float) * (2*h_halo_size + 2*v_halo_size));

    // LEFT
    for (int i = 0; i < counts[0] - 1; i++) {
      s_buffer[i + offsets[0]] = tile[i*h_halo_size + 1]; //(1,i), i~[0,v_halo)
    }
    // RIGHT
    for (int i = 0; i < counts[1] - 1; i++) {
      s_buffer[i + offsets[1]] = tile[(i+1)*h_halo_size - 2]; //(h_halo_size -2,i)
    }
    // DOWN
    for (int i = 0; i < counts[3] - 1; i++) {
      s_buffer[i + offsets[3]] = tile[(v_halo_size - 2)*h_halo_size + i]; //(i,v_halo_size -2)
    }
    // UP
    for (int i = 0; i < counts[2] - 1; i++) {
      s_buffer[i + offsets[2]] = tile[1*h_halo_size + i]; //(i,1)
    }
    

    /*MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], 
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], 
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)*/
    MPI_Neighbor_alltoallv(s_buffer, counts, offsets, MPI_FLOAT, r_buffer, counts, offsets, MPI_FLOAT, comcord);

    // LEFT
    if (coords[0] != 0 || 1) { //If not at grid left
      for (int i = 0; i < counts[0] - 1; i++) {
        tile[i*h_halo_size + 0] = r_buffer[i + offsets[0]]; //(0,i)
      }
    }
    // RIGHT
    if (coords[0] != dims[0] - 1 ||1) { //If not at grid right
      for (int i = 0; i < counts[1] - 1; i++) {
        tile[(i+1)*h_halo_size - 1] = r_buffer[i + offsets[1]]; //(h_halo_size -1,i)
      }
    }
    // DOWN
    if (coords[1] != dims[1] - 1 ||1) { //If not at grid bottom
      for (int i = 0; i < counts[3] - 1; i++) {
        tile[(v_halo_size - 1)*h_halo_size + i] = r_buffer[i + offsets[3]]; //(i,v_halo_size -1)
      }
    }
    // UP
    if (coords[1] != 0||1) { //If not at grid top
      for (int i = 0; i < counts[2] - 1; i++) {
        tile[0*h_halo_size + i] = r_buffer[i + offsets[2]]; //(i,0)
      }
    }
   
    free(s_buffer);
    free(r_buffer);
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  int delta = width - nx;
  int pos = 1 + width;
  for(int i = 1; i < nx + 1; i++){
    #pragma vector
    for(int j = 1; j < ny + 1; j++){
          tmp_image[pos] = image[pos] * 0.6f + (image[pos - 1] + image[pos + 1] + image[pos - width] + image[pos + width]) * 0.1f;
          pos += 1;
    }
  pos += delta;
  }
}



// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int i = 0; i < nx + 2; ++i) {
    for (int j = 0; j < ny + 2; ++j) {
      image[i + j * width] = 0.0;
      tmp_image[i + j * width] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int ib = 0; ib < nx; ib += tile_size) {
    for (int jb = 0; jb < ny; jb += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int i = ib + 1; i < ilim + 1; ++i) {
          for (int j = jb + 1; j < jlim + 1; ++j) {
            image[i + j * width] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int i = 1; i < nx + 1; ++i) {
    for (int j = 1; j < ny + 1; ++j) {
      if (image[i + j * width] > maximum) maximum = image[i + j * width];
    }
  }

  // Output image, converting to numbers 0-255
  for (int i = 1; i < nx + 1; ++i) {
    for (int j = 1; j < ny + 1; ++j) {
      fputc((char)(255.0 * image[i + j * width] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
