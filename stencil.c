#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h" 

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
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
  float* image = _mm_malloc(sizeof(float) * width * height, 64);
  float* tmp_image = _mm_malloc(sizeof(float) * width * height, 64);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  //sync
  MPI_Barrier(MPI_COMM_WORLD);

  
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

    int tile_width = ceil((width - 2)/dims[0]); // Default tile size, int col = dims[0];
    int tile_height = ceil((height -2)/dims[1]); // Default tile size, int row = dims[1];
    //Setup this tile
    float* tile = _mm_malloc(sizeof(float) * (tile_width + 2) * (tile_height + 2), 64);
    float* tmp_tile = _mm_malloc(sizeof(float) * (tile_width + 2) * (tile_height + 2), 64);
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
    if (coords[0] == dims[0] - 1) h_halo_size = nx - (dims[0]-1) * tile_width + 2; //Adjust for incomplete tile
    if (coords[1] == dims[1] - 1) v_halo_size = ny - (dims[1]-1) * tile_height + 2; //Adjust for incomplete tile
   
    //Copy from original image (incl. halo)
    for(int x_offset = 0 ; x_offset < tile_width + 2; x_offset++){
      for(int y_offset = 0; y_offset < tile_height + 2; y_offset++){
        tile[y_offset * tile_width + x_offset]     = image[(y_start + y_offset) * width + (x_start + x_offset)];
        tmp_tile[y_offset * tile_width + x_offset] = image[(y_start + y_offset) * width + (x_start + x_offset)];
      }
    }
    //Calculate neighbor positions (rank)
    int t_rank = coords[0] * dims[0] + coords[1];
    int neighbors[4] = {0,0,0,0};
    neighbors[0] = t_rank - dims[0];
    neighbors[1] = t_rank - 1;
    neighbors[2] = t_rank + 1;
    neighbors[3] = t_rank + dims[0];
    // Call the stencil kernel for iters
    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
      exchange(comcord, tile, h_halo_size, v_halo_size, coords, dims);
      stencil(nx, ny, width, height, tile, tmp_tile);
      exchange(comcord, tile, h_halo_size, v_halo_size, coords, dims);
      stencil(nx, ny, width, height, tmp_tile, tile);
    }
    double toc = wtime();

    //TODO: Merge and Output
    if(rank == MASTER){
      printf("------------------------------------\n");
      printf(" runtime: %lf s\n", toc - tic);
      printf("------------------------------------\n");

      output_image(OUTPUT_FILE, nx, ny, width, height, image);
    }

  }

  _mm_free(image);
  _mm_free(tmp_image);

  MPI_Finalize();
}

void exchange(MPI_Comm comcord, float * tile, int h_halo_size, int v_halo_size, int * coords, int * dims){
    //float * tile, int rows, int cols
    //Halo order: UP, DOWN, LEFT, RIGHT
    int counts[4] = {h_halo_size, h_halo_size, v_halo_size, v_halo_size};
    int offsets[4] = {0, h_halo_size, 2 * h_halo_size, 2 * h_halo_size + v_halo_size};
    int vdisp = 0;
    int hdisp = 0;

    float *s_buffer = _mm_malloc(sizeof(float) * (2*h_halo_size + 2*v_halo_size), 2*h_halo_size + 2*v_halo_size);
    float *r_buffer = _mm_malloc(sizeof(float) * (2*h_halo_size + 2*v_halo_size), 2*h_halo_size + 2*v_halo_size);

    if (coords[1] == dims[1]-1) vdisp = 0;
    else vdisp = 1;
    if (coords[0] == 0) hdisp = 0;
    else hdisp = 1;

    // UP
    for (int i = 1; i < counts[0] - 1; i++) {
      s_buffer[i + offsets[0]] = tile[1*h_halo_size + i]; //(i,1)
    }
    // DOWN
    for (int i = 1; i < counts[1] - 1; i++) {
      s_buffer[i + offsets[1]] = tile[(v_halo_size - 2)*h_halo_size + i]; //(i,v_halo_size -2)
    }
    // LEFT
    for (int i = 1; i < counts[2] - 1; i++) {
      s_buffer[i + offsets[2]] = tile[i*h_halo_size + 1]; //(1,i)
    }
    // RIGHT
    for (int i = 1; i < counts[3] - 1; i++) {
      s_buffer[i + offsets[3]] = tile[(i+1)*h_halo_size - 2]; //(h_halo_size -2,i)
    }
    

    /*MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], 
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], 
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)*/
    MPI_Neighbor_alltoallv(s_buffer, counts, offsets, MPI_FLOAT, r_buffer, counts, offsets, MPI_FLOAT, comcord);

    //TODO: Check Save in receive buffer
    // UP
    if (coords[1] != 0) { //If not at grid top
      for (int i = 1; i < counts[0] - 1; i++) {
        tile[0*h_halo_size + i] = s_buffer[i + offsets[0]]; //(i,0)
      }
    }
    // DOWN
    if (coords[1] != dims[1] - 1) { //If not at grid bottom
      for (int i = 1; i < counts[1] - 1; i++) {
        tile[(v_halo_size - 1)*h_halo_size + i] = s_buffer[i + offsets[1]]; //(i,v_halo_size -1)
      }
    }
    // LEFT
    if (coords[0] != 0) { //If not at grid left
      for (int i = 1; i < counts[2] - 1; i++) {
        tile[i*h_halo_size + 0] = s_buffer[i + offsets[2]]; //(0,i)
      }
    }
    // RIGHT
    if (coords[0] != dims[0] - 1) { //If not at grid right
      for (int i = 1; i < counts[3] - 1; i++) {
        tile[(i+1)*h_halo_size - 1] = s_buffer[i + offsets[3]]; //(h_halo_size -1,i)
      }
    }
   
    _mm_free(s_buffer);
    _mm_free(r_buffer);
}

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  int delta = height - ny;
  int pos = 1 + height;
  for(int i = 1; i < nx + 1; i++){
    #pragma vector
    for(int j = 1; j < ny + 1; j++){
          tmp_image[pos] = image[pos] * 0.6f + (image[pos - 1] + image[pos + 1] + image[pos - height] + image[pos + height]) * 0.1f;
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
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
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
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
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
