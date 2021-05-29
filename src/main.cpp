#include <stdio.h>
#include <string>
#include <iostream>
#include "ppm_parser.hh"
#include "../include/filter.hpp"
#include "../include/segdet.hpp"
// #include "parse.hh"

int main(int argc, char *argv[]){
    using namespace kalman;
    char *filename;
    if (argc == 2)
        filename = argv[1];
    //READ ppm file into image
    kalman::ppm_parser parser;

    kalman::image2d<uint8_t> img = parser.pgm(filename);


    auto out = kalman::detect_line(img, 10, 0);


    auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);

    labeled_arr(lab_arr, out);

    std::cout << "out";

    lab_arr.imsave("out.pgm");

    // //Parse It
    // auto width = vec[0];
    // auto height = vec[1];
    // vec.erase(vec.begin(), vec.begin() + 2);
    // kalman::parser parser;
    // auto parsed_vec = parser.parse(width, height, vec, 245);
    // for (vector<pair<int, int>> vec: parsed_vec){
    //     for (pair<int, int> pair: vec){
    //         cout << pair.first << "-" << pair.second << "  ";
    //     }
    //     cout << "\n";
    // }
    return 0;
}


// #include <iostream>
// #include "../include/filter.hpp"
// #include "../include/image2d.hpp"
// #include "../include/segdet.hpp"




// int main(int argc, char** argv)
// {
//     using namespace kalman;
//     auto img = image2d<int>(    {{1, 2, 3, 4},
//                                 {5, 6, 7, 8},
//                                 {7, 8, 9, 10}});
    
//     std::cout << img << "\nimg[1, 2] = " << img({1, 2}) << "\n";
//     img({1, 1}) = 42;

//     image_point pt = image_point(2, 2);
//     img(pt) = 69;

//     std::cout << img;

//     // for (auto elm : img.domain())
//     //     std::cout << elm << ";";


//     img.transform([](int value){ return value + 1;});

//     std::cout << img;

//     img.fill(42);
//     std::cout << img;



//     return 0;
// }