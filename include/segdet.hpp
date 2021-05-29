#pragma once

#include "segment.hpp"
#include "image2d.hpp"

#include <utility>

namespace kalman
{
  /**
   * Detects lines in the given image
   * @param image A ndbuffer representing the image to process (can be rgb or uint8)
   * @param min_len The minimum length of segments to detect
   * @param discontinuity The maximum accepted discontinuity for segments
   * @return A vector of detected segments
   */
  std::vector<Segment> detect_line(image2d<uint8_t> image, int min_len, int discontinuity);
  std::vector<Segment> detect_line(image2d<uint8_t>, int min_len, int discontinuity,
                                   const Parameters& params);

  /**
   * Draw segments in img out
   * @param out Image to draw in
   * @param segments
   */
  void labeled_arr(image2d<uint16_t> out, const std::vector<Segment>& segments);
} // namespace mln::contrib::segdet