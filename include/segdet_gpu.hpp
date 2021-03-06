#pragma once

#include "segment.hpp"
#include "matrix_tools_gpu.cuh"
#include <cinttypes>


struct Filter
{
    bool dead;
    bool isHorizontal;
    kalman_gpu::kMatrix<float, 4, 1> S;
    kalman_gpu::kMatrix<float, 4, 1> W;
    kalman_gpu::kMatrix<float, 3, 1> N;
    kalman_gpu::kMatrix<float, 4, 4> H;
    kalman_gpu::kMatrix<float, 4, 1> S_predicted;
    kalman_gpu::kMatrix<float, 3, 1> X_predicted;

    int obs_index;
    int obs_distance;

    int first_integration_col;
    int nb_integration;

    float sigma_position;
    float sigma_thickness;
    float sigma_luminosity;

    float sum_position;
    float sum_thickness;
    float sum_luminosity;

    float sum_sq_position;
    float sum_sq_thickness;
    float sum_sq_luminosity;

    float n_min;
    float n_max;

    Filter() = default;
    CUDA_CALLABLE_MEMBER Filter(float position, float thickness, float luminosity, int first_integ_col)
    : 
      dead(false),
      obs_index(-1),
      obs_distance(0),
      first_integration_col(first_integ_col),
      nb_integration(1),
      sigma_position(2),
      sigma_thickness(2),
      sigma_luminosity(57),
      sum_position(0),
      sum_thickness(0),
      sum_luminosity(0),
      sum_sq_position(0),
      sum_sq_thickness(0),
      sum_sq_luminosity(0),
      n_min(0),
      n_max(0)
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
std::vector<kalman::Segment> traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs);
