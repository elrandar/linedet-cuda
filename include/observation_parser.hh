#pragma once

#include <vector>
#include <utility>
#include <Eigen/Dense>

namespace kalman {
    class obs_parser {
    public:
        static std::pair<Eigen::Vector3d*, unsigned int*> parse_gpu(int width, int height, std::vector<u_int8_t> &img, int threshold);
        static std::pair<Eigen::Vector3d*, unsigned int*> parse_gpu2(int width, int height, std::vector<u_int8_t> &img, std::ptrdiff_t stride, int threshold);
        static std::pair<Eigen::Vector3d*, unsigned int*> parse_gpu3(int width, int height, std::vector<u_int8_t> &img, int threshold);
        std::vector<std::vector<Eigen::Vector3d>> parse(int width, int height, std::vector<u_int8_t> img, int threshold);
    };

    void test(void);
    void test_gpu(int*);
}
