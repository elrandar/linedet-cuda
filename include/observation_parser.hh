#pragma once
#include "matrix_tools.hpp"
#include <vector>

namespace kalman {
    class obs_parser {
        public:
            std::vector<std::vector<kMatrix<float>>> parse(int width, int height, std::vector<u_int8_t> img, int threshold) {
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
    };
}
