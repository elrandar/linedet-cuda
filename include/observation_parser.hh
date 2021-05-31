#include <vector>

namespace kalman {
    class obs_parser {
        public:
            std::vector<std::vector<std::pair<int, int>>> parse(int width, int height, std::vector<u_int8_t> img, int threshold) {
                std::vector<std::vector<std::pair<int, int>>> vec;
                for(int j = 0; j < width; j++){
                    std::vector<std::pair<int, int>> tmp_vec;
                    auto max = -1;
                    //auto pos_max = -1;
                    auto start = -1;
                    auto in_obs = false;
                    for(int i = 0; i < height; i++) {
                        if (img[i * width + j] < threshold) {
                            if (img[i * width + j] > max) {
                                max = img[i * width + j];
                            }
                            if (!in_obs) {
                                start = i;
                                in_obs = true;
                            }
                        }
                        else {
                            if (max != -1) {
                                tmp_vec.push_back({(start + i) / 2, i - start});
                            }
                            max = -1;
                            //pos_max = -1;
                            in_obs = false;
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
