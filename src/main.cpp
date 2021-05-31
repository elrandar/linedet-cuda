#include <stdio.h>
#include <string>
#include <iostream>
#include "../include/ppm_parser.hh"
#include "../include/filter.hpp"
#include "../include/segdet.hpp"
#include "../include/observation_parser.hh"
// #include "parse.hh"

int main(int argc, char *argv[])
{
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
        auto out = kalman::detect_line(img, 10, 0);

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);
     
        // Output
        lab_arr.imsave("out.pgm");
    }
    else
    {
        if (mode == "--parallel")
        {
            std::cout << "Processing Image in parallel, using CPU\n";

            kalman::obs_parser parser;
            auto parsed_vec = parser.parse(img.width, img.height, img.get_buffer(), 225);
            for (std::vector<std::pair<int, int>> vec: parsed_vec)
            {
                for (std::pair<int, int> pair: vec)
                {
                    std::cout << pair.first << "-" << pair.second << "  ";
                }
                std::cout << "\n";
            }
        }
        else if (mode == "--gpu")
        {
            std::cout << "Processing Image in parallel, using GPU\n";
        }
        else
        {
            throw std::invalid_argument("Unknown mode. Second argument can be '--parallel', '--gpu' or '--sequential'.");
        }
    }
    return 0;
}