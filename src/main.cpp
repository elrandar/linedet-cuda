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
        auto out = kalman::detect_line(img, 20, 0);

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

        auto out = kalman_batch::detect_line(img, 20, 0);

        // Image labellisation
        auto lab_arr = kalman::image2d<uint16_t>(img.width, img.height);
        labeled_arr(lab_arr, out);

        // Output
        lab_arr.imsave("out.pgm");
        */

        // Doing the exact same thing as gpu does
        std::cout << "USING CPU\n";
        std::vector<std::vector<kMatrix<float>>> res;
        auto parser = kalman::obs_parser();

        for (auto i = 0u; i < 100; ++i)
            res = parser.parse(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res[i].size() << " ";
        std::cout << "\n";
    }
    else if (mode == "--gpu")
    {
        std::pair<obs_elem*, unsigned int*> observations = kalman::obs_parser::parse_gpu(img.width, img.height, img.get_buffer(), 225);
        
        std::vector<float> observations_array{};
        std::vector<int> obs_per_col_array{0};

        int col_obs_sum = 0;
        for (int i = 0; i < img.width; i++)
        {
            col_obs_sum += observations.second[i];
            obs_per_col_array.push_back(col_obs_sum);
        }

        for (int i = 0; i < img.width; i++)
        {
            for (int j = 0; j < observations.second[i]; j++)
            {
                observations_array.push_back(observations.first[j * img.width + i].position);
                observations_array.push_back(observations.first[j * img.width + i].thickness);
                observations_array.push_back(observations.first[j * img.width + i].luminosity);
            }
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
    else if (mode == "--gpu_obs")
    {
        std::cout << "USING GPU\n";
        std::pair<obs_elem *, unsigned int*> res;

        res = kalman::obs_parser::parse_gpu(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res.second[i] << " ";
        std::cout << "\n";

        delete[] res.first;
        delete[] res.second;
    }
    else if (mode == "--gpu2")
    {
        std::cout << "USING GPU\n";
        auto stride = img.width * sizeof(uint8_t);
        std::pair<obs_elem *, unsigned int*> res;

        res = kalman::obs_parser::parse_gpu2(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res.second[i] << " ";
        std::cout << "\n";

        delete[] res.first;
        delete[] res.second;
    }
    else if (mode == "--gpu3")
    {
        std::cout << "USING GPU\n";
        std::pair<obs_elem *, unsigned int*> res;

        res = kalman::obs_parser::parse_gpu3(img.width, img.height, img.get_buffer(), 245);

        std::cout << "Sizes:\n";
        for (auto i =0u; i < img.width; ++i)
            std::cout << res.second[i] << " ";
        std::cout << "\n";

        delete[] res.first;
        delete[] res.second;
    }
    else
        throw std::invalid_argument("Unknown mode. Second argument can be '--parallel', '--gpu' or '--sequential'.");

    return 0;
}
