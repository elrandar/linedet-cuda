#include <stdio.h>
#include <string>
#include <iostream>
#include "../include/ppm_parser.hh"
#include "../include/filter.hpp"
#include "../include/segdet.hpp"
#include "../include/observation_parser.hh"
#include "../include/matrix_tools.hpp"
#include "../include/segdet_batch.hpp"
#include "../include/segdet_gpu.hpp"

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

        auto out = kalman_batch::detect_line(img, 20, 0);

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);
    
        // Output
        lab_arr.imsave("out.pgm");
    }
    else if (mode == "--gpu")
    {
        auto p = obs_parser();
        auto observations = p.parse(img.width, img.height, img.get_buffer_const(), 225);
        
        std::vector<float> observations_array{};
        std::vector<int> obs_per_col_array{0};

        int col_obs_sum = 0;
        for (auto col : observations)
        {
            for (auto obs : col)
            {
                col_obs_sum += 1;
                observations_array.push_back(obs(0, 0));
                observations_array.push_back(obs(1, 0));
                observations_array.push_back(obs(2, 0));
            }
            obs_per_col_array.push_back(col_obs_sum);
        }

        std::cout << "Processing Image in parallel, using GPU\n";
        

        auto out = traversal_gpu(observations_array.data(),
                      obs_per_col_array.data(),
                      img.width,
                      img.height / 2,
                      observations_array.size() / 3);
        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);
    
        // Output
        lab_arr.imsave("out.pgm");
    }
    else
        throw std::invalid_argument("Unknown mode. Second argument can be '--parallel', '--gpu' or '--sequential'.");

    
    return 0;
}