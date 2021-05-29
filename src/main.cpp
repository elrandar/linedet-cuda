#include <stdio.h>
#include <string>
#include <iostream>
#include "file.hh"
#include "parse.hh"

int main(int argc, char *argv[]){
    char *filename;
    if (argc == 2)
        filename = argv[1];
    //READ file
    kalman::file file;
    auto vec = file.ppm(filename);
    //Parse It
    auto width = vec[0];
    auto height = vec[1];
    vec.erase(vec.begin(), vec.begin() + 2);
    kalman::parser parser;
    auto parsed_vec = parser.parse(width, height, vec, 245);
    for (vector<pair<int, int>> vec: parsed_vec){
        for (pair<int, int> pair: vec){
            cout << pair.first << "-" << pair.second << "  ";
        }
        cout << "\n";
    }
    return 0;
}


#include <iostream>
#include "../include/filter.hpp"
#include "../include/image2d.hpp"




int main(int argc, char** argv)
{

    auto img = image2d<int>(    {{1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {7, 8, 9, 10}});
    
    std::cout << img << "\nimg[1, 2] = " << img({1, 2}) << "\n";
    img({1, 1}) = 42;

    point pt = point(2, 2);
    img(pt) = 69;

    std::cout << img;

    for (auto elm : img.domain())
        std::cout << elm << ";";
    return 0;
}