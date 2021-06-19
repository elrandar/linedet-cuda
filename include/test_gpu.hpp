#pragma once

#include "matrix_tools_gpu.cuh"
#include <cinttypes>


struct Filter
{
    bool isHorizontal;
    kalman_gpu::kMatrix<float, 4, 1> S;
    kalman_gpu::kMatrix<float, 4, 1> W;
    kalman_gpu::kMatrix<float, 3, 1> N;
    kalman_gpu::kMatrix<float, 4, 4> H;
    // float S[4]; // 4 x 1
    // float W[4]; // 4 x 1
    // float N[3]; // 3 x 1
    // float H[16]; // 4 x 4
    kalman_gpu::kMatrix<float, 4, 1> S_predicted;
    kalman_gpu::kMatrix<float, 3, 1> X_predicted;

    int obs_index;
    int obs_distance;

    int nb_integration;

    float sigma_position;
    float sigma_thickness;
    float sigma_luminosity;
    Filter() = default;
    CUDA_CALLABLE_MEMBER Filter(float position, float thickness, float luminosity)
    : 
      obs_index(-1),
      obs_distance(0),
      nb_integration(1),
      sigma_position(2),
      sigma_thickness(2),
      sigma_luminosity(57)
      {
        S(0, 0) = position;
        S(1, 0) = 0;
        S(2, 0) = thickness;
        S(3, 0) = luminosity;
        for (int i = 0; i < 4; i++)
          W.buffer[i] = 0;        
        for (int i = 0; i < 3; i++)
          N.buffer[i] = 0;
        for (int i = 0; i < 16; i++)
          H.buffer[i] = 0;
        H.buffer[0] = 1;
        H.buffer[5] = 1;
        H.buffer[10] = 1;
        H.buffer[15] = 1;
      }
};

void test_gpu(uint8_t* hostBuffer, int width, int height);
void traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs);
