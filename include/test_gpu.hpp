#pragma once
#include <cinttypes>


struct Filter
{
    bool isHorizontal;
    float S[4]; // 4 x 1
    float W[4]; // 4 x 1
    float N[3]; // 3 x 1
    float H[16]; // 4 x 4
    float S_predicted[4];
    float X_predicted[3];

    int obs_index;
    int obs_distance;

    int nb_integration;

    float sigma_position;
    float sigma_thickness;
    float sigma_luminosity;
    Filter() = default;
    Filter(float position, float thickness, float luminosity)
    : 
      obs_index(-1),
      obs_distance(0),
      nb_integration(1),
      sigma_position(2),
      sigma_thickness(2),
      sigma_luminosity(57)
      {
        S[0] = position;
        S[1] = 0;
        S[2] = thickness;
        S[3] = luminosity;
        for (int i = 0; i < 4; i++)
          W[i] = 0;        
        for (int i = 0; i < 3; i++)
          N[i] = 0;
        for (int i = 0; i < 16; i++)
          H[i] = 0;
        H[0] = 1;
        H[5] = 1;
        H[10] = 1;
        H[15] = 1;
      }
};

void test_gpu(uint8_t* hostBuffer, int width, int height);
void traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs);
