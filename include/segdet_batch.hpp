#pragma once

#include "../include/segment.hpp"
#include "../include/image2d.hpp"

namespace kalman_batch
{
    using namespace kalman;
    /**
 * Detects lines in the given image
 * @param image A ndbuffer representing the image to process (can be rgb or uint8)
 * @param min_len The minimum length of segments to detect
 * @param discontinuity The maximum accepted discontinuity for segments
 * @return A vector of detected segments
 */
    std::vector <Segment> detect_line_batch(image2d <uint8_t> &image, int min_len, int discontinuity);
}
