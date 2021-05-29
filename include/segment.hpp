#pragma once

#include "../include/parameter.hpp"
#include <cassert>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mln
{
  namespace contrib
  {
    namespace segdet
    {
      struct Point
      {
        Point(u_int32_t x, u_int32_t y, u_int32_t size)
          : x(x)
          , y(y)
          , thickness(size)
        {
        }
        Point(u_int32_t n, u_int32_t t, u_int32_t size, bool is_horizontal)
          : thickness(size)
        {
          if (is_horizontal)
          {
            x = t;
            y = n;
          }
          else
          {
            x = n;
            y = t;
          }
        }

        u_int32_t x;
        u_int32_t y;
        u_int32_t thickness;
      };

      struct Segment
      {
        Segment(std::vector<Point> points_vector, const std::vector<Point>& underOther, float firstPartSlope,
                float lastPartSlope, bool isHorizontal)
          : first_point(points_vector[0])
          , last_point(points_vector[points_vector.size() - 1])
          , points(std::move(points_vector))
          , first_part_slope(firstPartSlope)
          , last_part_slope(lastPartSlope)
          , length(1 + (isHorizontal ? points[points.size() - 1].x - points[0].x
                                     : points[points.size() - 1].y - points[0].y))
          , is_horizontal(isHorizontal)
        {
          nb_pixels = 0;

          for (auto& machin : points)
            nb_pixels += machin.thickness;

          for (auto& machin : underOther)
          {
            nb_pixels += machin.thickness;
            points.push_back(machin);
          }
        }


        Point first_point;
        Point last_point;

        std::vector<Point> points;

        float     first_part_slope;
        float     last_part_slope;
        u_int32_t length;
        u_int32_t nb_pixels;
        bool      is_horizontal;
      };
    } // namespace segdet
  }   // namespace contrib
} // namespace mln
