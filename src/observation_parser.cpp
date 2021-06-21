#include <iostream>
#include "observation_parser.hh"

namespace kalman {

    void test(void) {
        int *a = new int[100];
        for (auto i = 0u; i < 100; ++i) {
            std::cout << a[i] << ' ';
        }
        std::cout << '\n';

        test_gpu(a);

        for (auto i = 0u; i < 100; ++i) {
            std::cout << a[i] << ' ';
        }
        std::cout << '\n';

        delete[] a;
    }

}
