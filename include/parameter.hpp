#pragma once

#define GET_VARIABLE_NAME(Variable) (#Variable)
#include <cstdlib>
#include <map>
#include <string>

namespace mln
{
  namespace contrib
  {
    namespace segdet
    {
      struct Parameters
      {
        Parameters()                  = default;
        u_int32_t nb_values_to_keep   = 30;
        u_int32_t variance_position   = 2;
        u_int32_t variance_thickness  = 1;
        u_int32_t variance_luminosity = 12;

        u_int32_t default_sigma_position   = 2;
        u_int32_t default_sigma_thickness  = 2;
        u_int32_t default_sigma_luminosity = 57;

        u_int32_t min_nb_values_sigma  = 10;
        double    sigma_pos_min        = 1;
        double    sigma_thickness_min  = 0.64;
        double    sigma_luminosity_min = 13;

        double    slope_max_vertical     = 1.05;
        double    slope_max_horizontal   = 1.0;
        u_int32_t max_llum               = 225;
        u_int32_t max_thickness          = 100;
        double    ratio_lum              = 1;
        double    merge_slope_variation  = 0.4;
        double    merge_distance_max     = 8;
        u_int32_t max_slopes_too_large   = 5;
        double    threshold_intersection = 0.8;
        u_int32_t minimum_for_fusion     = 15;

        explicit Parameters(const std::map<std::string, double>& map)
        {
          for (auto& kvp : map)
          {
            auto str = kvp.first;
            auto val = kvp.second;

            if (str == GET_VARIABLE_NAME(nb_values_to_keep))
              nb_values_to_keep = val;
            else if (str == GET_VARIABLE_NAME(variance_position))
              variance_position = val;
            else if (str == GET_VARIABLE_NAME(variance_thickness))
              variance_thickness = val;
            else if (str == GET_VARIABLE_NAME(variance_luminosity))
              variance_luminosity = val;
            else if (str == GET_VARIABLE_NAME(default_sigma_position))
              default_sigma_position = val;
            else if (str == GET_VARIABLE_NAME(default_sigma_thickness))
              default_sigma_thickness = val;
            else if (str == GET_VARIABLE_NAME(default_sigma_luminosity))
              default_sigma_luminosity = val;
            else if (str == GET_VARIABLE_NAME(min_nb_values_sigma))
              min_nb_values_sigma = val;
            else if (str == GET_VARIABLE_NAME(sigma_pos_min))
              sigma_pos_min = val;
            else if (str == GET_VARIABLE_NAME(sigma_thickness_min))
              sigma_thickness_min = val;
            else if (str == GET_VARIABLE_NAME(sigma_luminosity_min))
              sigma_luminosity_min = val;
            else if (str == GET_VARIABLE_NAME(slope_max_vertical))
              slope_max_vertical = val;
            else if (str == GET_VARIABLE_NAME(slope_max_horizontal))
              slope_max_horizontal = val;
            else if (str == GET_VARIABLE_NAME(max_thickness))
              max_thickness = val;
            else if (str == GET_VARIABLE_NAME(max_llum))
              max_llum = val;
            else if (str == GET_VARIABLE_NAME(ratio_lum))
              ratio_lum = val;
            else if (str == GET_VARIABLE_NAME(merge_slope_variation))
              merge_slope_variation = val;
            else if (str == GET_VARIABLE_NAME(merge_distance_max))
              merge_distance_max = val;
            else if (str == GET_VARIABLE_NAME(max_slopes_too_large))
              max_slopes_too_large = val;
            else if (str == GET_VARIABLE_NAME(threshold_intersection))
              threshold_intersection = val;
            else if (str == GET_VARIABLE_NAME(minimum_for_fusion))
              minimum_for_fusion = val;
            else
              exit(-255);
          }
        }

        bool is_valid() const
        {
          return nb_values_to_keep > minimum_for_fusion && max_slopes_too_large < nb_values_to_keep;
        }
      };
    } // namespace segdet
  }   // namespace contrib
} // namespace mln



