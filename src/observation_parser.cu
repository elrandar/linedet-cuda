#include "../include/observation_parser.hh"

#include <host_defines.h>
#include <device_launch_parameters.h>

__global__ void test_kernel(int* a, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;;
    a[i] += 1;
}
