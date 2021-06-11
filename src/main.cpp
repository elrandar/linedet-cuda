#include <stdio.h>
#include <string>
#include <iostream>
#include "../include/ppm_parser.hh"
#include "../include/filter.hpp"
#include "../include/segdet.hpp"
#include "../include/observation_parser.hh"
#include "../include/matrix_tools.hpp"
#include "../include/segdet_gpu.hpp"

int main(int argc, char *argv[])
{
//
//    std::cout << compute_det(std::vector<double>({9, 0, 5,
//    8, 4, 5,
//    7, 3, 8}), 3);
//
//    for (auto & elm : invert_matrix(std::vector<double>({9, 0, 5,
//                                                         8, 4, 5,
//                                                         7, 3, 8}), 3))
//    {
//    std::cout << elm << " ";
//    }
//
//    std::cout << '\n' << std::endl;
//    auto mat = Eigen::Matrix<double, 3, 3>();
//    mat << 9, 0, 5,
//            8, 4, 5,
//            7, 3, 8;
//    std::cout << invert_matrix3(mat
//    );
    using namespace kalman;
    char *filename;
    std::string mode = "--sequential";

    // Argument parsing

    if (argc == 2)
        filename = argv[1];
    else if (argc == 3)
    {
        mode = argv[1];
        filename = argv[2];
    }
    else
    {
        std::cout << "Usage :\n kalman-gpu --[gpu / sequential / parallel] <img>\n";
        return 0;
    }


    //READ ppm file into image
    kalman::ppm_parser parser;

    kalman::image2d<uint8_t> img = parser.pgm(filename);


    if (mode == "--sequential")
    {
        std::cout << "Processing Image Sequentially\n";
        // Sequential Line detection
        auto out = kalman::detect_line(img, 20, 0);

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);
     
        // Output
        lab_arr.imsave("out.pgm");
    }
    else if (mode == "--batch")
    {
        std::cout << "Processing Image in batches, using CPU\n";

        auto out = kalman_gpu::detect_line(img, 20, 0);

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);
    
        // Output
        lab_arr.imsave("out.pgm");
    }
    else if (mode == "--gpu")
    {
        std::cout << "Processing Image in parallel, using GPU\n";
    }
    else
        throw std::invalid_argument("Unknown mode. Second argument can be '--parallel', '--gpu' or '--sequential'.");

    
    return 0;
}