#pragma once
#include "matrix_tools.hpp"
#include <vector>
#include <utility>

namespace kalman {
    class obs_parser {
    public:
        static std::pair<kMatrix<double>*, unsigned int*> parse_gpu(int width, int height, std::vector<u_int8_t> &img, int threshold);
        static std::pair<kMatrix<double>*, unsigned int*> parse_gpu2(int width, int height, std::vector<u_int8_t> &img, std::ptrdiff_t stride, int threshold);
        static std::pair<kMatrix<double>*, unsigned int*> parse_gpu3(int width, int height, std::vector<u_int8_t> &img, int threshold);
        std::vector<std::vector<kMatrix<double>>> parse(int width, int height, std::vector<u_int8_t> img, int threshold);
    };

    void test(void);
    void test_gpu(int*);
}
