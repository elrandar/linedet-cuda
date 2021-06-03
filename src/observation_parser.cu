#include <iostream>
#include <cassert>
#include <err.h>
#include "observation_parser.hh"
#include "observation_parser.cuh"

namespace kalman {

    std::vector<std::vector<Eigen::Vector3d>>
    obs_parser::parse(int width, int height, std::vector<u_int8_t> img, int threshold) {
        std::vector<std::vector<Eigen::Vector3d>> vec;

        for (int j = 0; j < width; ++j) {
            std::vector<Eigen::Vector3d> tmp_vec;
            auto max = -1;
            auto start = -1;
            auto in_obs = false;
            for (int i = 0; i < height; ++i) {
                if (img[i * width + j] < threshold) {
                    if (img[i * width + j] > max) {
                        max = img[i * width + j];
                    }
                    if (!in_obs) {
                        start = i;
                        in_obs = true;
                    }
                }
                else {
                    if (max != -1) {
                        tmp_vec.emplace_back(static_cast<double>((start + i) / 2),
                                           static_cast<double>(i - start),
                                           static_cast<double>(max));
                    }
                    max = -1;
                    in_obs = false;
                }
            }
            if (max != -1)
                tmp_vec.emplace_back(static_cast<double>((start + height) / 2),
                                   static_cast<double>(height - start),
                                   static_cast<double>(max));
            vec.emplace_back(tmp_vec);
        }

        return vec;
    }

    __global__ void test_kernel(int* a) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        a[i] = 1;
    }

    std::vector<std::vector<Eigen::Vector3d>>
    parse_gpu(int width, int height, std::vector<u_int8_t> img, int threshold) {
        std::vector<std::vector<Eigen::Vector3d>> res;

        return res;
    }

    void test_gpu(int *a) {
        int *a_device;

        auto err = cudaMalloc(&a_device, 100 * sizeof(int));
        if (err)
            errx(1, "Cuda malloc error code %d", err);

        cudaMemcpy(a_device, a, 100 * sizeof(int), cudaMemcpyHostToDevice);

        test_kernel<<<100, 1>>>(a_device);

        cudaMemcpy(a, a_device, 100 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(a_device);
    }
}
