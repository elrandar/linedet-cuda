#include <iostream>
#include "../include/filter.hpp"
#include "../include/image2d.hpp"
#include "../include/segdet.hpp"




int main(int argc, char** argv)
{
    using namespace kalman;
    auto img = image2d<int>(    {{1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {7, 8, 9, 10}});
    
    std::cout << img << "\nimg[1, 2] = " << img({1, 2}) << "\n";
    img({1, 1}) = 42;

    image_point pt = image_point(2, 2);
    img(pt) = 69;

    std::cout << img;

    // for (auto elm : img.domain())
    //     std::cout << elm << ";";


    img.transform([](int value){ return value + 1;});

    std::cout << img;

    img.fill(42);
    std::cout << img;

    return 0;
}