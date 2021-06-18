#pragma once
#include <cinttypes>

void test_gpu(uint8_t* hostBuffer, int width, int height);
void traversal_gpu(float* obsHostBuffer, int* obsCount, int width, int max_height, int nb_obs);
