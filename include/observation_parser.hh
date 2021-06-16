#pragma once
#include "matrix_tools.hpp"
#include <vector>
#include <utility>


namespace kalman {
    struct obs_elem {
    double position;
    double thickness;
    double luminosity;
    obs_elem() = default;
    obs_elem(double a, double b, double c)
    : position(a), thickness(b), luminosity(c)
    {}
};
    class obs_parser {
    public:
        static std::pair<obs_elem*, unsigned int*> parse_gpu(int width, int height, std::vector<u_int8_t> &img, int threshold);
        // static std::pair<obs_elem*, unsigned int*> parse_gpu2(int width, int height, std::vector<u_int8_t> &img, std::ptrdiff_t stride, int threshold);
        // static std::pair<obs_elem*, unsigned int*> parse_gpu3(int width, int height, std::vector<u_int8_t> &img, int threshold);
        std::vector<std::vector<kMatrix<double>>> parse(int width, int height, std::vector<u_int8_t> img, int threshold);
    };

    void test(void);
    void test_gpu(int*);
}
