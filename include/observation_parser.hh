#pragma once

#include <vector>
#include <Eigen/Dense>

namespace kalman {
    class obs_parser {
    public:
        static std::vector<std::vector<Eigen::Vector3d>> parse(int width, int height, std::vector<u_int8_t> img, int threshold);
        std::vector<std::vector<Eigen::Vector3d>> parse_gpu(int width, int height, std::vector<u_int8_t> img, int threshold);
    };

    void test(void);
    void test_gpu(int*);
}