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
