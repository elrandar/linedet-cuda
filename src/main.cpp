#include <stdio.h>
#include <string>
#include <iostream>
#include "ppm_parser.hh"
#include "filter.hpp"
#include "segdet.hpp"
#include "observation_parser.hh"

#include <iostream>

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
        auto out = kalman::detect_line(img, 10, 0, "seq");

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);

        // Output
        lab_arr.imsave("out.pgm");
    }
    else if (mode == "--batch")
    {
        /* OLD CODE
        std::cout << "Processing Image in batches, using CPU\n";

        auto out = kalman::detect_line(img, 10, 0, "batch");

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);

        // Output
        lab_arr.imsave("out.pgm");
        */

        // Doing the exact same thing as gpu does
        std::cout << "USING CPU\n";
        auto res = kalman::obs_parser().parse(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res[i].size() << " ";
        std::cout << "\n";
    }
    else if (mode == "--gpu")
    {
        std::cout << "USING GPU\n";
        auto res = kalman::obs_parser::parse_gpu(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res.second[i] << " ";
        std::cout << "\n";
    }
    else
        throw std::invalid_argument("Unknown mode. Second argument can be '--parallel', '--gpu' or '--sequential'.");

    return 0;
}
