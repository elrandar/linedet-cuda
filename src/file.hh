#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <algorithm>

using namespace std;
namespace kalman {
    class file {
        // Return vector
        // Vector[0] = width
        // Vector[1] = height
        // Vector[n] = grayscale value at n-2
        public:
        vector<int> ppm(string name) {
            vector<int> vec;

            string line;
            ifstream myfile;
            myfile.open(name);

            if(!myfile.is_open()) {
                perror("Error opening the file");
                exit(-1);
            }
            std::getline(myfile, line); // P3
            std::getline(myfile, line);

            auto tmp_vec = this->parse_width_height(line);
            vec.insert(vec.end(), tmp_vec.begin(), tmp_vec.end());
            std::getline(myfile, line); // 255
            while(std::getline(myfile, line)) {
                auto tmp_vec = this->parse_width_height(line);
                vec.insert(vec.end(), tmp_vec.begin(), tmp_vec.end());
            }
            myfile.close();
            return vec;
        }
        vector<int> parse_width_height(string line) {
            vector<int> vec;
            char *token = strtok(&(line[0]) , " ");
            while (token != NULL)
            {
                vec.push_back(stoi(token));
                token = strtok(NULL, " ");
            }
            return vec;
        }

        vector<int> parse_line(string line) {
            vector<int> vec;
            char *token = strtok(&(line[0]) , " ");
            while (token != NULL)
            {
                vec.push_back(stoi(token));
                token = strtok(NULL, " ");
            }
            return vec;
        }
    };
}
