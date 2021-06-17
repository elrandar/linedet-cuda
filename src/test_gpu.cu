#include "test_gpu.hpp"
#include <cassert>
#include <iostream>


[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  std::cout << msg << " (" << fname << ", line: " << line << ")\n";
  std::cout << "Error " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << "\n";
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void mykernel(unsigned char* buffer, int width, int height, size_t pitch, 
                         unsigned char* outBuffer, size_t out_pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    outBuffer[out_pitch * y + x] = (x + buffer[pitch * y + x]) / 2;
}


void test_gpu(uint8_t* hostBuffer, int width, int height)
{
    cudaError_t rc = cudaSuccess;

    // alloc device memory
    unsigned char* devInputBuffer;
    size_t in_pitch;
    unsigned char* devOutputBuffer;
    size_t out_pitch;

    rc = cudaMallocPitch(&devInputBuffer, &in_pitch, width * sizeof(uint8_t), height);
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMallocPitch(&devOutputBuffer, &out_pitch, width * sizeof(uint8_t), height);
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMemcpy2D(devInputBuffer, in_pitch, hostBuffer, width*sizeof(uint8_t), width * sizeof(uint8_t),
        height, cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");

    // run the kernel with blocks of size 64 * 64

    
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    std::cout << "max nb_threads is " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "max threads dim is : " << prop.maxThreadsDim[0] << ", "
                                         << prop.maxThreadsDim[1] << ", "
                                         << prop.maxThreadsDim[2] << std::endl;
 
    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    std::cout << "running kernel of size " << w << " , " << h << std::endl;
    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    mykernel<<<dimGrid, dimBlock>>>(devInputBuffer, width, height,
                                    in_pitch, devOutputBuffer, out_pitch);
    if (cudaPeekAtLastError())
        abortError("Computation Error");
    

    rc = cudaMemcpy2D(hostBuffer, width * sizeof(uint8_t), devOutputBuffer, out_pitch, width * sizeof(char), height, cudaMemcpyDeviceToHost);
      if (rc)
    abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(devInputBuffer);
    if (rc)
        abortError("Unable to free memory");
    rc = cudaFree(devOutputBuffer);
    if (rc)
        abortError("Unable to free memory");
}

__global__ void update_filters(float* obs_buffer, int* obs_count, int col, int max_height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int nb_obs_in_col = obs_count[col + 1] - obs_count[col];
    if (x >= nb_obs_in_col)
        return;

    for (int i = 0; i < nb_obs_in_col; i++)
    {
    printf("col n%d, obs n%d, it has position of %f, thickness of %f and lum of %f\n",
            col,
            i, obs_buffer[obs_count[col] + i * 3],
            obs_buffer[obs_count[col] + i * 3 + 1],
            obs_buffer[obs_count[col] + i * 3 + 2]);
    }
}

void traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs)
{
    cudaError_t rc = cudaSuccess;

    // alloc device memory
    float* obs_buffer;
    int* obs_count_buffer;


    rc = cudaMalloc(&obs_buffer, nb_obs * sizeof(float) * 3);
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMalloc(&obs_count_buffer, width * sizeof(int));
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMemcpy(obs_buffer, obsHostBuffer, nb_obs * sizeof(float) * 3,
                    cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");
    rc = cudaMemcpy(obs_count_buffer, obsCount, width * sizeof(int),
            cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");

    // rc = cudaMalloc()

    // run the kernel with blocks of size 64 * 64

    
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    std::cout << "max nb_threads is " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "max threads dim is : " << prop.maxThreadsDim[0] << ", "
                                         << prop.maxThreadsDim[1] << ", "
                                         << prop.maxThreadsDim[2] << std::endl;
 
    int bsize = 512;
    int h = std::ceil((float)max_height / (bsize));

    std::cout << "running kernel of size " << h << std::endl;

    for (int i = 1; i < width; i++)
    {
        update_filters<<<h, bsize>>>(obs_buffer, obs_count_buffer, i,
                                            max_height);
    }
   
    if (cudaPeekAtLastError())
        abortError("Computation Error");
    

    // rc = cudaMemcpy2D(hostBuffer, width * sizeof(uint8_t), devOutputBuffer, out_pitch, width * sizeof(char), height, cudaMemcpyDeviceToHost);
    //   if (rc)
    // abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(obs_buffer);
    if (rc)
        abortError("Unable to free memory");
    rc = cudaFree(obs_count_buffer);
    if (rc)
        abortError("Unable to free memory");
}