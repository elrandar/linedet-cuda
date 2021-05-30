#include <vector>

namespace kalman {
    class parser {
        public:
            std::vector<std::vector<std::pair<int, int>>> parse(int width, int height, std::vector<u_int8_t> img, int threshold) {
                std::vector<std::vector<std::pair<int, int>>> vec;
                for(int j = 0; j < width; j++){
                    std::vector<std::pair<int, int>> tmp_vec;
                    auto max = -1;
                    //auto pos_max = -1;
                    auto start = -1;
                    auto previous = false;
                    for(int i = 0; i < height; i++) {
                        if (img[i * width + j] < threshold) {
                            if (previous) {
                                if (img[i * width + j] > max) {
                                    max = img[i * width + j];
                                    //pos_max = i;
                                }
                            }
                            else {
                                start = i;
                                previous = true;
                            }
                        }
                        else {
                            if (max != -1) {
                                /*
                                 * method not suitable for big lines
                                if (pos_max - start  < i) {
                                    tmp_vec.push_back({pos_max, pos_max - start});
                                }
                                else {
                                    tmp_vec.push_back({pos_max, i - pos_max});
                                }
                                */
                                tmp_vec.push_back({(start + i) / 2, i - start});
                            }
                            max = -1;
                            //pos_max = -1;
                            previous = false;
                        }
                    }
                    if (max != -1)
                        tmp_vec.push_back({(start + height) / 2, height - start});
                    vec.push_back(tmp_vec);
                }
                return vec;
            }
    };
}
