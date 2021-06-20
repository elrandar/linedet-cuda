#include <iostream>
#include <cassert>
#include <err.h>
#include "matrix_tools.hpp"
#include "observation_parser.hh"
#include "observation_parser.cuh"

namespace kalman {


    // - - - - - LINE BY LINE PARSING WITH CACHE - - - - -

    struct cache_obj {
        int max;
        int start;
        bool in_obs;
    };

    __global__ void set_cache(cache_obj *cache, unsigned int size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
            return;

        cache[i] = {-1, -1, false};
    }

    /*
        for(int j = 0; j < width; j++){
            std::vector<kMatrix<float>> tmp_vec;
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
                        tmp_vec.push_back(kMatrix<float>({static_cast<float>((start + i) / 2),
                                            static_cast<float>(i - start),
                                            static_cast<float>(max)}, 3, 1));
                    }
                    max = -1;
                    //pos_max = -1;
                    in_obs = false;
                }
            }
            if (max != -1)
                tmp_vec.push_back(kMatrix<float>({static_cast<float>((start + height) / 2),
                                    static_cast<float>(height - start),
                                    static_cast<float>(max)}, 3, 1));
            vec.push_back(tmp_vec);
        }
    */

    __global__ void last_line_parser(int width, int height, obs_elem *vec, cache_obj *cache, unsigned int *sizes)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width)
            return;

        auto *cparams = cache + i;

        if (cparams->max != -1)
        {
            auto res_index = sizes[i] * width + i;
            vec[res_index].position = static_cast<float>((cparams->start + height) / 2);
            vec[res_index].thickness =  static_cast<float>(height - cparams->start);
            vec[res_index].luminosity = static_cast<float>(cparams->max);
            ++(sizes[i]);
        }
    }

    __global__ void line_parser(int width, u_int8_t *img, obs_elem *vec, int threshold, cache_obj *cache, unsigned int line_number, unsigned int *sizes)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width)
            return;

        auto *cparams = cache + i;
        auto val_at = img[i];

        if (val_at < threshold) {
            if (val_at > cparams->max) {
                cparams->max = val_at;
            }
            if (!(cparams->in_obs)) {
                cparams->start = line_number;
                cparams->in_obs = true;
            }
        }
        else
        {
            if (cparams->max != -1) {
                auto res_index = sizes[i] * width + i;
                vec[res_index].position = static_cast<float>((cparams->start + line_number) / 2);
                vec[res_index].thickness =  static_cast<float>(line_number - cparams->start);
                vec[res_index].luminosity = static_cast<float>(cparams->max);
                sizes[i] += 1;
            }
            cparams->max = -1;
            cparams->in_obs = false;
        }
    }

    std::pair<obs_elem*, unsigned int*>
        obs_parser::parse_gpu3(int width, int height, std::vector<u_int8_t> &img_host, int threshold)
        {
            auto size = width * height;

            // Allocating memory for result array
            obs_elem *vec = nullptr;
            auto err = cudaMalloc(&vec, sizeof(obs_elem) * size / 2);
            if (err)
                errx(1, "Cuda vec malloc error code %d", err);

            // Allocating memory for image
            u_int8_t *img = nullptr;
            err = cudaMalloc(&img, sizeof(u_int8_t) * size);
            if (err)
                errx(1, "Cuda img malloc error code %d", err);

            // Copying image into gpu buffer
            err = cudaMemcpy(img, (u_int8_t*)(&img_host[0]), size * sizeof(u_int8_t), cudaMemcpyHostToDevice);
            if (err)
                errx(1, "Cuda img-host memcpy error code %d", err);

            // Allocating res sizes for each column
            unsigned int *sizes = nullptr;
            err = cudaMalloc(&sizes, sizeof(unsigned int) * width);
            if (err)
                errx(1, "Cuda sizes malloc error code %d", err);
            err = cudaMemset(sizes, 0, sizeof(unsigned int) * width);
            if (err)
                errx(1, "Cuda sizes memset error code %d", err);

            cache_obj *cache;
            err = cudaMalloc(&cache, sizeof(cache_obj) * width);
            if (err)
                errx(1, "Cuda cache malloc error code %d", err);

            auto bsize = 32; // Do not change this number, else program won't work
            int w = std::ceil(static_cast<float>(width) / bsize);
            int h = 1;
            dim3 dimBlock(bsize, bsize);
            dim3 dimGrid(w, h);

            // - - - - - - ALGO - - - - - -

            set_cache<<<dimGrid, dimBlock>>>(cache, width);

            // Calling kernel *width* times, *heigh* times
            for (auto line_number = 0u; line_number < height; ++line_number)
                line_parser<<<dimGrid, dimBlock>>>(width, img + (line_number * width), vec, threshold, cache, line_number, sizes);

            last_line_parser<<<dimGrid, dimBlock>>>(width, height, vec, cache, sizes);

            // - - - - - - - - - - - END ALGO - - - - - - - - - -

            // Creating result array
            obs_elem *res = new obs_elem[size / 2];
            // Copying res onto CPU
            err = cudaMemcpy(res, vec, (size / 2) * sizeof(obs_elem), cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda vec-host memcpy error code %d", err);

            // Creating result sizes array
            unsigned int *host_sizes = new unsigned int[width];
            // Copying res sizes onto CPU
            err = cudaMemcpy(host_sizes, sizes, width * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda sizes-host memcpy error code %d", err);

            // Freeing dev buffers
            err = cudaFree(vec);
            if (err)
                errx(1, "Cuda vec free error code %d", err);
            err = cudaFree(img);
            if (err)
                errx(1, "Cuda img free error code %d", err);
            err = cudaFree(sizes);
            if (err)
                errx(1, "Cuda sizes free error code %d", err);
            err = cudaFree(cache);
            if (err)
                errx(1, "Cuda cache free error code %d", err);

            return std::make_pair(res, host_sizes);
        }

    // - - - - - COLUMN BY COLUMN PARSING - - - - -

    __global__ void column_parser(int width, int height, u_int8_t *img, obs_elem *vec, int threshold, unsigned int *sizes)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width)
            return;

        auto vec_size = i;

        auto max = -1;
        auto start = -1;
        auto in_obs = false;

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
                    vec[vec_size].position = static_cast<float>((start + j) / 2);
                    vec[vec_size].thickness =  static_cast<float>(j - start);
                    vec[vec_size].luminosity = static_cast<float>(max);
                    vec_size += width;
                }
                max = -1;
                in_obs = false;
            }
        }

        if (max != -1) {
            vec[vec_size].position = static_cast<float>((start + height) / 2);
            vec[vec_size].thickness = static_cast<float>(height - start);
            vec[vec_size].luminosity = static_cast<float>(max);
            vec_size += width;
        }


        sizes[i] = (vec_size - i) / width;
    }


    std::pair<obs_elem *, unsigned int*>
        obs_parser::parse_gpu(int width, int height, std::vector<u_int8_t> &img_host, int threshold)
        {
            auto size = width * height;

            // Allocating memory for result array
            obs_elem *vec = nullptr;
            auto err = cudaMalloc(&vec, sizeof(obs_elem) * size / 2);
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

            auto bsize = 32; // Do not change this number, else program won't work
            int w = std::ceil(static_cast<float>(width) / bsize);
            int h = 1;
            dim3 dimBlock(bsize, bsize);
            dim3 dimGrid(w, h);

            // Calling kernel *width* times
            column_parser<<<dimGrid, dimBlock>>>(width, height, img, vec, threshold, sizes);

            // Creating result array
            obs_elem *res = new obs_elem[size / 2];
            // Copying res onto CPU
            err = cudaMemcpy(res, vec, size * sizeof(obs_elem) / 2, cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda vec-host memcpy error code %d", err);

            // Creating result sizes array
            unsigned int *host_sizes = new unsigned int[width];
            // Copying res sizes onto CPU
            err = cudaMemcpy(host_sizes, sizes, width * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda sizes-host memcpy error code %d", err);

            // Freeing dev buffers
            err = cudaFree(vec);
            if (err)
                errx(1, "Cuda vec free error code %d", err);
            err = cudaFree(img);
            if (err)
                errx(1, "Cuda img free error code %d", err);
            err = cudaFree(sizes);
            if (err)
                errx(1, "Cuda sizes free error code %d", err);

            return std::make_pair(res, host_sizes);
        }

    // - - - - - COLUMN BY COLUMN PARSING WITH PITCH - - - - -

    __global__ void column_parser_pitched(int width, int height, size_t pitch_img, size_t pitch_vec, u_int8_t *img, obs_elem *vec, int threshold, unsigned int *sizes)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width)
            return;

        auto vec_size = i;

        auto max = -1;
        auto start = -1;
        auto in_obs = false;

        for (int j = 0; j < height; ++j) {
            auto index = j * pitch_img + i;
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
                    vec[vec_size].position = static_cast<float>((start + j) / 2);
                    vec[vec_size].thickness =  static_cast<float>(j - start);
                    vec[vec_size].luminosity = static_cast<float>(max);
                    vec_size += pitch_vec;
                }
                max = -1;
                in_obs = false;
            }
        }

        if (max != -1) {
            vec[vec_size].position = static_cast<float>((start + height) / 2);
            vec[vec_size].thickness = static_cast<float>(height - start);
            vec[vec_size].luminosity = static_cast<float>(max);
            vec_size += pitch_vec;
        }

        sizes[i] = (vec_size - i) / pitch_vec;
    }

    std::pair<obs_elem*, unsigned int*>
        obs_parser::parse_gpu2(int width, int height, std::vector<u_int8_t> &img_host, int threshold)
        {
            auto vec_width = width * sizeof(obs_elem);
            auto vec_height = height / 2 + 1;

            // Allocating memory for result array
            size_t pitch_vec = 0;
            obs_elem *vec = nullptr;
            auto err = cudaMallocPitch(&vec, &pitch_vec, vec_width, vec_height);
            if (err)
                errx(1, "Cuda vec malloc error code %d", err);

            // Allocating memory for image
            size_t pitch_img = 0;
            u_int8_t *img = nullptr;
            err = cudaMallocPitch(&img, &pitch_img, sizeof(u_int8_t) * width, height);
            if (err)
                errx(1, "Cuda img malloc error code %d", err);

            // Allocating res sizes for each column
            unsigned int *sizes = nullptr;
            err = cudaMalloc(&sizes, sizeof(unsigned int) * width);
            if (err)
                errx(1, "Cuda sizes malloc error code %d", err);

            // Copying image into gpu buffer
            err = cudaMemcpy2D(img, pitch_img, (u_int8_t*)(&img_host[0]), width * sizeof(u_int8_t), width * sizeof(u_int8_t), height, cudaMemcpyHostToDevice);
            if (err)
                errx(1, "Cuda img-host memcpy error code %d", err);

            auto bsize = 32; // Do not change this number, else program won't work
            int w = std::ceil(static_cast<float>(width) / bsize);
            int h = 1;
            dim3 dimBlock(bsize, bsize);
            dim3 dimGrid(w, h);

            // Calling kernel *width* times
            column_parser_pitched<<<dimGrid, dimBlock>>>(width, height, pitch_img / sizeof(u_int8_t), pitch_vec / sizeof(obs_elem), img, vec, threshold, sizes);
            if (cudaPeekAtLastError())
                errx(1, "Computation Error");

            // Creating result array
            obs_elem *res = new obs_elem[width * vec_height];
            // Copying res onto CPU
            err = cudaMemcpy2D(res, vec_width, vec, pitch_vec, vec_width, vec_height, cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda vec-host memcpy error code %d", err);

            // Creating result sizes array
            unsigned int *host_sizes = new unsigned int[width];
            // Copying res sizes onto CPU
            err = cudaMemcpy(host_sizes, sizes, width * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (err)
                errx(1, "Cuda sizes-host memcpy error code %d", err);

            // Freeing dev buffers
            err = cudaFree(vec);
            if (err)
                errx(1, "Cuda vec free error code %d", err);
            err = cudaFree(img);
            if (err)
                errx(1, "Cuda img free error code %d", err);
            err = cudaFree(sizes);
            if (err)
                errx(1, "Cuda sizes free error code %d", err);

            return std::make_pair(res, host_sizes);
        }

    // - - - - - CPU PARSING - - - - -

    std::vector<std::vector<kMatrix<float>>> obs_parser::parse(int width, int height, std::vector<u_int8_t> img, int threshold)
    {
        std::vector<std::vector<kMatrix<float>>> vec;
        for(int j = 0; j < width; j++){
            std::vector<kMatrix<float>> tmp_vec;
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
                        tmp_vec.push_back(kMatrix<float>({static_cast<float>((start + i) / 2),
                                    static_cast<float>(i - start),
                                    static_cast<float>(max)}, 3, 1));
                    }
                    max = -1;
                    //pos_max = -1;
                    in_obs = false;
                }
            }
            if (max != -1)
                tmp_vec.push_back(kMatrix<float>({static_cast<float>((start + height) / 2),
                            static_cast<float>(height - start),
                            static_cast<float>(max)}, 3, 1));
            vec.push_back(tmp_vec);
        }
        return vec;
    }

}
