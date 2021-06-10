#include <iostream>
#include <cassert>
#include <err.h>
#include "observation_parser.hh"
#include "observation_parser.cuh"

namespace kalman {

/*
    {
        auto max = -1;
        auto start = -1;
        auto in_obs = false;

        if (max != -1)
            tmp_vec.emplace_back(static_cast<double>((start + height) / 2),
                                static_cast<double>(height - start),
                                static_cast<double>(max));
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
                                        static_cast<double>(unsigned i - start),
                                        static_cast<double>(max));
                }
                max = -1;
                in_obs = false;
            }
        }
        vec.emplace_back(tmp_vec);
    }
    */

    __global__ void test_kernel(int* a) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        a[i] = 1;
    }

    __global__ void column_parser(int width, int height, u_int8_t *img, Eigen::Vector3d *vec, int threshold, unsigned int *sizes)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width)
            return;

        auto vec_size = i;

        auto max = -1;
        auto start = -1;
        auto in_obs = false;

        if (max != -1) {
            vec[vec_size] = Eigen::Vector3d(static_cast<double>((start + height) / 2),
                                         static_cast<double>(height - start),
                                         static_cast<double>(max));
            vec_size += width;
        }

        for (int j = 0; j < height; ++j) {
            auto index = j * width + i;
            auto val_at = img[index];

            if (val_at < threshold) {
                if (val_at > max) {
                    max = val_at;
                }
                if (!in_obs) {
                    start = j;
                    in_obs = true;
                }
            }
            else {
                if (max != -1) {
                    vec[vec_size] = Eigen::Vector3d(static_cast<double>((start + j) / 2),
                                        static_cast<double>(j - start),
                                        static_cast<double>(max));
                    vec_size += width;
                }
                max = -1;
                in_obs = false;
            }
        }

        sizes[i] = (vec_size - i) / width;
    }

    std::pair<Eigen::Vector3d *, unsigned int*>
    obs_parser::parse_gpu(int width, int height, std::vector<u_int8_t> &img_host, int threshold)
    {
        auto size = width * height;
        // TODO is it better to gow with a vecotr of array of size height ??
        // TODO Try inplace with one array
        // TODO Try line by line
        // TODO Check pitch

        // Allocating memory for result array
        Eigen::Vector3d *vec = nullptr;
        auto err = cudaMalloc(&vec, sizeof(Eigen::Vector3d) * size);
        if (err)
            errx(1, "Cuda vec malloc error code %d", err);

        // Allocating memory for image
        u_int8_t *img = nullptr;
        err = cudaMalloc(&img, sizeof(u_int8_t) * size);
        if (err)
            errx(1, "Cuda img malloc error code %d", err);

        // Allocating res sizes for each column
        unsigned int *sizes = nullptr;
        err = cudaMalloc(&sizes, sizeof(unsigned int) * width);
        if (err)
            errx(1, "Cuda sizes malloc error code %d", err);

        // Copying image into gpu buffer
        err = cudaMemcpy(img, (u_int8_t*)(&img_host[0]), size * sizeof(u_int8_t), cudaMemcpyHostToDevice);
        if (err)
            errx(1, "Cuda img-host memcpy error code %d", err);

        // Getting gpu specs
        int devId = 0;
        // There may be more devices!
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devId);
        int xThreads = deviceProp.maxThreadsDim[0];
        dim3 DimBlock(xThreads, 1, 1);// 1D VecAddint
        auto xBlocks = (int) ceil(width / xThreads);
        dim3 DimGrid(xBlocks, 1, 1);

        // Calling kernel *width* times
        column_parser<<<1024, 1024>>>(width, height, img, vec, threshold, sizes);

        // Creating result array
        Eigen::Vector3d *res = new Eigen::Vector3d[size];
        // Copying res onto CPU
        err = cudaMemcpy(res, vec, size * sizeof(Eigen::Vector3d), cudaMemcpyDeviceToHost);
        if (err)
            errx(1, "Cuda vec-host memcpy error code %d", err);

        // Creating result sizes array
        unsigned int *host_sizes = new unsigned int[width];
        // Copying res sizes onto CPU
        err = cudaMemcpy(host_sizes, sizes, width * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (err)
            errx(1, "Cuda sizes-host memcpy error code %d", err);

        std::cout << "Parse with GPU done.\n";

        return std::make_pair(res, host_sizes);
    }

    std::vector<std::vector<Eigen::Vector3d>> obs_parser::parse(int width, int height, std::vector<u_int8_t> img, int threshold)
    {
        std::vector<std::vector<Eigen::Vector3d>> vec;
        for(int j = 0; j < width; j++){
            std::vector<Eigen::Vector3d> tmp_vec;
            auto max = -1;
            //auto pos_max = -1;
            auto start = -1;
            auto in_obs = false;
            for(int i = 0; i < height; i++) {
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
                        tmp_vec.push_back({static_cast<double>((start + i) / 2),
                                            static_cast<double>(i - start),
                                            static_cast<double>(max)});
                    }
                    max = -1;
                    //pos_max = -1;
                    in_obs = false;
                }
            }
            if (max != -1)
                tmp_vec.push_back({static_cast<double>((start + height) / 2),
                                    static_cast<double>(height - start),
                                    static_cast<double>(max)});
            vec.push_back(tmp_vec);
        }
        return vec;
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
