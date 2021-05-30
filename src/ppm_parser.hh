#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include "../include/image2d.hpp"

namespace kalman {
    class ppm_parser {
        // Return vector
        // Vector[0] = width
        // Vector[1] = height
        // Vector[n] = grayscale value at n-2
        public:
        image2d<uint8_t> pgm(std::string filename) {
            std::vector<uint8_t> vec;

            std::string line;
            std::ifstream myfile;
            myfile.open(filename);

            if(!myfile.is_open()) {
                perror("Error opening the file");
                exit(-1);
            }

            std::getline(myfile, line); // FILE FORMAT

            if (line != "P2")
            {
                if (line == "P5")
                    throw std::invalid_argument("The image file format should be ASCII PGM (P2), not binary PGM (P5).");
                if (line == "P6" || line == "P3")
                    throw std::invalid_argument("The image file format should be ASCII PGM (P2), not PPM (P3 or P6).");
                else
                    throw std::invalid_argument("Invalid input image format. The image file format should be ASCII PGM (P2).");
            }

            std::getline(myfile, line);

            auto width_height_vec = parse_width_height(line);

            int width = width_height_vec[0];
            int height = width_height_vec[1];

            std::getline(myfile, line); // 255

            while(std::getline(myfile, line)) {
                auto tmp_vec = parse_line(line);
                vec.insert(vec.end(), tmp_vec.begin(), tmp_vec.end());
            }
            myfile.close();

            auto output_image = image2d<uint8_t>(width, height);
            output_image.set_buffer(vec);

            return output_image;
        }
        std::vector<uint16_t> parse_width_height(std::string line) {
            std::vector<uint16_t> vec;
            char *token = strtok(&(line[0]) , " ");
            while (token != NULL)
            {
                vec.push_back(std::stoi(token));
                token = strtok(NULL, " ");
            }
            return vec;
        }

        std::vector<uint8_t> parse_line(std::string line) {
            std::vector<uint8_t> vec;
            char *token = strtok(&(line[0]) , " ");
            while (token != NULL)
            {
                vec.push_back(std::stoi(token));
                token = strtok(NULL, " ");
            }
            return vec;
        }
    };
}
