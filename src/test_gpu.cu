#include "test_gpu.hpp"
#include "matrix_tools_gpu.cuh"

// #include "segment_gpu.cuh"
#include <cassert>
#include <vector>
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


__device__ void predict(Filter* f)
  {
    using namespace kalman_gpu;

    float a[16] = {1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    kMatrix<float, 4, 4> A = kMatrix<float, 4, 4>(a);
    float a_t[16] = {1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    kMatrix<float, 4, 4> A_transpose(a_t);
    float c[12] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    kMatrix<float, 3, 4> C(c);
    float c_t[12] = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
    kMatrix<float, 4, 3> C_transpose(c_t);
    float vn[9] = {2, 0, 0, 0,
                   1, 0, 0, 0, 12};
    kMatrix<float, 3, 3> Vn(vn);

    
    f->S_predicted = kMatrix<float, 4, 1>();
    matmul(A, f->S, f->S_predicted);
    add(f->S_predicted, f->W, f->S_predicted);

    f->X_predicted = kMatrix<float, 3, 1>();
    matmul(C, f->S_predicted, f->X_predicted);
    add(f->X_predicted, f->N, f->X_predicted);


    // f.S_predicted = A * f.S + f.W;
    // f.X_predicted = C * f.S_predicted + f.N;

    float thik_d2 = f->X_predicted(1, 0) / 2;
    f->n_min = f->X_predicted(0, 0) - thik_d2;
    f->n_max = f->X_predicted(0, 0) + thik_d2;

    matmul(f->H, A_transpose, f->H);
    matmul(A, f->H, f->H);

    // f.H = A * f.H * A_transpose;

    f->W.buffer[0] = 0;
    f->W.buffer[1] = 0;

    f->obs_index = -1;
  }

__device__  bool accepts_sigma(float prediction, float observation, float sigma)
  {
    // printf("prediction : %f, observation : %f\n", prediction, observation);
    if (prediction > observation)
      return (prediction - observation) <= 3 * sigma;
    return (observation - prediction) <= 3 * sigma;
  }

__device__  bool accepts(Filter* f, float* obs, float min, float max)
  {
    if (f->nb_integration > 10 && obs[1] / f->X_predicted(1, 0) > 1.5 &&
        std::abs(obs[1] - f->X_predicted(1, 0)) > 3)
    {
      return false;
    }

    if (f->n_max < min || max < f->n_min)
      return false;
    
    // printf("prediction vector : %f, %f, %f | obs_vector : %f, %f, %f\n", f->X_predicted(0,0), f->X_predicted(1,0), f->X_predicted(2,0),
    //         obs[0], obs[1], obs[2]);
    return accepts_sigma(f->X_predicted(0, 0), obs[0], f->sigma_position) &&
           accepts_sigma(f->X_predicted(1, 0), obs[1], f->sigma_thickness) &&
           accepts_sigma(f->X_predicted(2, 0), obs[2], f->sigma_luminosity);
  }

__device__ bool find_match(Filter* f, float* obs_ptr)
{
    float obs_thick = obs_ptr[1];
    float obs_thick_d2 = obs_thick / 2;

    float obs_n_min = obs_ptr[0] - obs_thick_d2;
    if (obs_n_min != 0)
        obs_n_min--;
    float obs_n_max = obs_ptr[0] + obs_thick_d2 + 1;


    // TODO under other part here when everything works 

    return accepts(f, obs_ptr, obs_n_min, obs_n_max);
}


__device__ void choose_nearest(Filter* f, float* obs_ptr, int obs_id)
{
    float distance = std::abs(obs_ptr[0] - f->X_predicted(0, 0));

    if (f->obs_index == -1 || distance < f->obs_distance)
    {
        f->obs_index = obs_id;
        f->obs_distance = distance;
    }
}

__device__ void insert_into_filters_list(Filter* f, int* integrations, int col, float* obs_ptr)
{
    integrations[col] = f->obs_index;
    f->nb_integration += 1;

    float position = obs_ptr[0];
    float thickness = obs_ptr[1];
    float luminosity = obs_ptr[2];

    f->sum_position += position;
    f->sum_sq_position += position * position;
    f->sum_thickness += thickness;
    f->sum_sq_thickness += thickness * thickness;
    f->sum_luminosity += luminosity;
    f->sum_sq_luminosity += luminosity * luminosity;
    // f.slopes.push_back(compute_slope(f));

    // if (f.n_values.size() > params.nb_values_to_keep)
    // {
    //   auto thick = f.thicknesses[0];
    //   auto nn    = f.n_values[0];
    //   auto tt    = f.t_values[0];

    //   f.thicknesses.erase(f.thicknesses.begin());
    //   f.t_values.erase(f.t_values.begin());
    //   f.n_values.erase(f.n_values.begin());
    //   f.luminosities.erase(f.luminosities.begin());
    //   f.slopes.erase(f.slopes.begin());

    //   if (f.first_slope == std::nullopt)
    //     f.first_slope = std::make_optional(f.slopes[f.slopes.size() - 1]);

    //   f.segment_points.emplace_back(nn, tt, thick, f.is_horizontal);
    // }
  }

 __device__ void integrate(Filter* f, int t, float* obs_ptr, int* integrations)
  {
    using namespace kalman_gpu;

    float c[12] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    kMatrix<float, 3, 4> C(c);
    float c_t[12] = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
    kMatrix<float, 4, 3> C_transpose(c_t);
    float vn[9] = {2, 0, 0, 0,
                   1, 0, 0, 0, 12};
    kMatrix<float, 3, 3> Vn(vn);

    kMatrix<float, 3, 1> observation{};
    observation.buffer[0] = obs_ptr[0];
    observation.buffer[1] = obs_ptr[1];
    observation.buffer[2] = obs_ptr[2];

    // if (!f.currently_under_other.empty())
    // {
    //   for (auto& elm : f.currently_under_other)
    //     f.under_other.push_back(elm);
    //   f.currently_under_other.clear();
    // }

    kMatrix<float, 4, 3> G{};
    {
        kMatrix<float, 4, 3> tmp_res{};
        matmul(f->H, C_transpose, tmp_res);
        kMatrix<float, 3, 3> tmp_res_2{};
        matmul(C, tmp_res, tmp_res_2);
        add(tmp_res_2, Vn, tmp_res_2);
        invert_matrix3(tmp_res_2, tmp_res_2);

        matmul(C_transpose, tmp_res_2, tmp_res);

        matmul(f->H, tmp_res, G);
    } // auto G = f.H * C_transpose * invert_matrix3(C * f.H * C_transpose + Vn);

    {
        kMatrix<float, 3, 1> obs_diff{};
        subtract(observation, f->X_predicted, obs_diff);
        matmul(G, obs_diff, f->S);
        add(f->S_predicted, f->S, f->S);
    } // f.S    = f.S_predicted + G * (observation - f.X_predicted);


    {
        float id4_buf[16] = {1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1};
        auto id4 = kMatrix<float, 4, 4>(id4_buf);
        kMatrix<float, 4, 4> tmp_res{};
        matmul(G, C, tmp_res);
        subtract(id4, tmp_res, tmp_res);
        kMatrix<float, 4, 4> tmp_res_2{};
        matmul(tmp_res, f->H, tmp_res_2);

        for (int i = 0; i < 16; i++)
            f->H.buffer[i] = tmp_res_2.buffer[i];
    } // f.H    = (id4 - G * C) * f.H;

    insert_into_filters_list(f, integrations, t, obs_ptr);

    // auto   length = f.slopes.size();
    // double second_derivative =
    //     (f.slopes[length - 1] - f.slopes[length - 2]) / (f.t_values[length - 1] - f.t_values[length - 2]);
    // f.W(0, 0)          = 0.5 * second_derivative;
    // f.W(1, 0)          = second_derivative;
    // f.last_integration = t;
  }

__device__ float std_calc(float sum_sq, float sum, float n)
{
    float mean_xx = sum_sq / n;
    float mean = sum / n;
    return sqrt(mean_xx - mean * mean); 
}

__device__ void compute_sigmas(Filter* f)
{
    int n = f->nb_integration;
    if (n > 10)
    {
      f->sigma_position   = std_calc(f->sum_sq_position, f->sum_position, n) + 1;
      f->sigma_thickness  = std_calc(f->sum_sq_thickness, f->sum_thickness, n) * 2 + 0.64;
      f->sigma_luminosity = std_calc(f->sum_sq_luminosity, f->sum_luminosity, n) + 13;
    }
}

__device__ bool filter_has_to_continue(Filter* f)
{
    return false;
}

__global__ void update_filters(float* obs_buffer, int* obs_count, int col, int max_height,
                               Filter* filter_buffer, int* integrations_buffer, int integration_padding,
                               int* obs_used_buffer)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int nb_obs_in_col = obs_count[col + 1] - obs_count[col];

    if (x >= max_height)
        return;


    Filter* f = filter_buffer + x;
    if (f->dead)
        return;

    int* integrations = integrations_buffer + (x * integration_padding);

    predict(f);

    // kalman_gpu::print(f->S);
    // kalman_gpu::print(f->X);

    for (int i = 0; i < nb_obs_in_col; i++)
    {
        float* obs_ptr = obs_buffer + obs_count[col - 1] + i * 3;
        bool accepted = find_match(f, obs_ptr);

        if (accepted)
        {
            choose_nearest(f, obs_ptr, i);
        }
    }

    
    if (f->obs_index != -1)
    {
        // set obs matched in obs match array
        obs_used_buffer[f->obs_index] = 1;
        integrate(f, col, obs_buffer + obs_count[col - 1] + f->obs_index * 3,
                          integrations);
        compute_sigmas(f);
    }
    else if (filter_has_to_continue(f))
    {
        for (int i = 0; i < 4; i++)
            f->S(i, 0) = f->S_predicted(i, 0);
    }
    else
    {
        f->dead = true;
    }
    // for (int i = 0; i < nb_obs_in_col; i++)
    // {
    // printf("col n%d, obs n%d, it has position of %f, thickness of %f and lum of %f\n",
    //         col,
    //         i, obs_buffer[obs_count[col] + i * 3],
    //         obs_buffer[obs_count[col] + i * 3 + 1],
    //         obs_buffer[obs_count[col] + i * 3 + 2]);
    // }
}

int get_max(int* buffer, int size)
{
    int max = buffer[0];
    for (int i = 1; i < size; i++)
    {
        if (buffer[i] - buffer[i - 1] > max)
            max = buffer[i];
    }
    return max;
}

void traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs)
{
    cudaError_t rc = cudaSuccess;

    // alloc device memory
    float* obs_buffer;
    int* obs_count_buffer;
    int* obs_used_buffer;

    int max_observations_col = get_max(obsCount, width);

    int integration_padding = width;
    std::vector<Filter> filter_host_buffer = std::vector<Filter>(nb_obs);
    std::vector<int> integrations_host_buffer = std::vector<int>(nb_obs * width, -1);
    std::vector<int> obs_used_host_buffer = std::vector<int>(max_observations_col, 0);
    int* integrations_device_buffer;
    Filter* filter_device_buffer;
    int nb_active_filters;

    rc = cudaMalloc(&obs_buffer, nb_obs * sizeof(float) * 3);
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMalloc(&obs_count_buffer, width * sizeof(int));
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMalloc(&obs_used_buffer, max_observations_col * sizeof(int));
    if (rc)
        abortError("Fail buffer alloc");
    rc = cudaMalloc(&filter_device_buffer, nb_obs * sizeof(Filter));
    if (rc)
        abortError("Cuda Malloc fail");
    rc = cudaMalloc(&integrations_device_buffer, nb_obs * width * sizeof(int));
    if (rc)
        abortError("Cuda Malloc fail");
    

    for (int i = 0; i < obsCount[1]; i++)
    {
        filter_host_buffer[i] = Filter(obsHostBuffer[i * 3],
                                       obsHostBuffer[i * 3 + 1],
                                       obsHostBuffer[i * 3 + 2],
                                       0);
        integrations_host_buffer[i * width] = i;
    }
    nb_active_filters = obsCount[1];

    rc = cudaMemcpy(obs_buffer, obsHostBuffer, nb_obs * sizeof(float) * 3,
                cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");
    rc = cudaMemcpy(obs_count_buffer, obsCount, width * sizeof(int),
            cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");
    rc = cudaMemcpy(filter_device_buffer, filter_host_buffer.data(), nb_obs * sizeof(Filter),
        cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");
    rc = cudaMemcpy(integrations_device_buffer, integrations_host_buffer.data(), nb_obs * width * sizeof(int),
        cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");
    rc = cudaMemcpy(obs_used_buffer, obs_used_host_buffer.data(), max_observations_col * sizeof(int),
                cudaMemcpyHostToDevice);
    if (rc)
        abortError("Cpy host to device fail");

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties( &prop, 0);

    // std::cout << "max nb_threads is " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "max threads dim is : " << prop.maxThreadsDim[0] << ", "
    //                                      << prop.maxThreadsDim[1] << ", "
    //                                      << prop.maxThreadsDim[2] << std::endl;
 


    for (int i = 1; i < width; i++)
    {
        // int i = 1;
        int bsize = 512;
        int h = std::ceil((float)nb_active_filters / (bsize));

        std::cout << "running kernel of size " << h << std::endl;

        update_filters<<<h, bsize>>>(obs_buffer, obs_count_buffer, i,
                                            nb_active_filters, filter_device_buffer,
                                            integrations_device_buffer, integration_padding,
                                            obs_used_buffer);

        
        // copy back obs_used_buffer, integrations_device_buffer, filter_device_buffer

        if (cudaPeekAtLastError())
            abortError("Computation Error");


        rc = cudaMemcpy(filter_host_buffer.data(), filter_device_buffer, nb_obs * sizeof(Filter),
            cudaMemcpyDeviceToHost);
        if (rc)
            abortError("Cpy host to device fail");
        rc = cudaMemcpy(obs_used_host_buffer.data(), obs_used_buffer, max_observations_col * sizeof(int),
            cudaMemcpyDeviceToHost);
        if (rc)
            abortError("Cpy host to device fail");

        // create filters from unused observations
        int nb_obs_in_col = obsCount[i + 1] - obsCount[i];

        for (int j = 0; j < nb_obs_in_col; j++)
        {
            float* obs_ptr = obsHostBuffer + obsCount[i - 1] + j * 3;
            if (obs_used_host_buffer[j] == 0)
            {
                filter_host_buffer[nb_active_filters] = Filter(obs_ptr[0],
                                obs_ptr[1],
                                obs_ptr[2],
                                i);
                nb_active_filters++;
            }
            else
                obs_used_host_buffer[j] = 0;
        }

        rc = cudaMemcpy(obs_used_buffer, obs_used_host_buffer.data(), max_observations_col * sizeof(int),
                cudaMemcpyHostToDevice);
        rc = cudaMemcpy(filter_device_buffer, filter_host_buffer.data(), nb_obs * sizeof(Filter),
                cudaMemcpyHostToDevice);
        if (rc)
            abortError("Cpy host to device fail");
        if (rc)
            abortError("Cpy host to device fail");
    }
    

    rc = cudaMemcpy(integrations_host_buffer.data(), integrations_device_buffer, nb_obs * width * sizeof(int),
        cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Cpy host to device fail");

    // Free
    rc = cudaFree(obs_buffer);
    if (rc)
        abortError("Unable to free memory");
    rc = cudaFree(obs_count_buffer);
    if (rc)
        abortError("Unable to free memory");
    rc = cudaFree(filter_device_buffer);
    if (rc)
        abortError("Cuda Free fail");
    rc = cudaFree(integrations_device_buffer);
    if (rc)
        abortError("Cuda Free fail");
}