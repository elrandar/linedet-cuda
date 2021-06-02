#include <cuda_runtime.h>
#include "../include/observation_parser.hh"

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


    std::vector<std::vector<Eigen::Vector3d>>
    obs_parser::parse_gpu(int width, int height, std::vector<u_int8_t> img, int threshold) {
        std::vector<std::vector<Eigen::Vector3d>> res;
        return res;
    }
}
