
#include <Eigen/Dense>
#include <algorithm>
#include <utility>

#include "../include/filter.hpp"
#include "../include/linearregression.hpp"
#include "../include/observation_parser.hh"
#include "../include/segdet.hpp"


namespace kalman
{
    /**
   * Give the value of the pixel in (n, t) according to traversal direction
   * @param image Image to take pixel from
   * @param n
   * @param t
   * @param is_horizontal
   * @return the value of the pixel
   */
    uint8_t image_at(image2d<uint8_t> image, int n, int t, bool is_horizontal)
    {
        // TODO remove the check done by image.at(,) using image(,)
        return is_horizontal ? image.at({t, n}) : image.at({n, t});
    }


    /**
   * Determine the observation Matrix
   * @param image
   * @param n
   * @param t
   * @param n_max
   * @param is_horizontal
   * @return Observation Eigen matrix
   */
    Eigen::Matrix<double, 3, 1> determine_observation(const image2d<uint8_t> &image, uint32_t &n, uint32_t t,
                                                      uint32_t n_max, bool is_horizontal, Parameters params)
    {
        uint32_t thickness = 0;
        uint32_t n_max_lum = 0;

        std::vector<uint8_t> luminosities_list = std::vector<uint8_t>();
        uint32_t lumi;

        // n + thickess: current position in the n-th line
        while (n + thickness < n_max && (lumi = image_at(image, n + thickness, t, is_horizontal)) < params.max_llum)
        {
            luminosities_list.push_back(lumi);

            if (lumi < luminosities_list[n_max_lum])
                n_max_lum = thickness;

            thickness += 1;
        }

        uint32_t n_to_skip = n + thickness;              // Position of the next n to work on
        uint32_t max_lum = luminosities_list[n_max_lum]; // Max luminosity of the current span

        // m_lum : max_luminosity of what is accepted in the span
        uint32_t m_lum = max_lum + (params.max_llum - max_lum) * params.ratio_lum;
        auto n_start = n;               // n_start is AT LEAST n
        uint32_t n_end = n + thickness; // n_end is AT MOST n + thickness

        if (n_end == n_max) // In case we stopped because of outOfBound value
            n_end--;

        while (luminosities_list[n - n_start] > m_lum)
            n += 1;

        while (image_at(image, n_end, t, is_horizontal) > m_lum)
            n_end--;

        n_end++;

        thickness = n_end - n;
        uint32_t position = n + thickness / 2;

        if (n_end - n > luminosities_list.size())
        {
            thickness--;
            n_end--;
            position = n + thickness / 2;
        }
        const double mean_val = mean(luminosities_list, n - n_start, n_end - n_start);

        n = n_to_skip; // Setting reference value of n

        return Eigen::Matrix<double, 3, 1>(position, thickness, mean_val);
    }

    /**
   * Say if a value is betwen two other
   * @param min
   * @param value
   * @param max
   * @return true if the value is between
   */
    inline bool in_between(double min, double value, double max) { return min <= value && value < max; }

    /**
   * Add a point in the under other attribute of filter
   * @param f Filter in which add the point
   * @param t
   * @param n
   * @param thickness
   */
    void add_point_under_other(Filter &f, uint32_t t, uint32_t n, uint32_t thickness)
    {
        if (f.is_horizontal)
            f.currently_under_other.emplace_back(t, n, thickness);
        else
            f.currently_under_other.emplace_back(n, t, thickness);
    }

    /**
   * Compute list of filters that accept the current Observation, add the filter inside accepted list
   * @param filters Current
   * @param obs Observation to match
   * @param t Current t
   * @param index Current index in n column
   */
    void find_match(std::vector<Filter> &filters, std::vector<Filter *> &accepted, const Eigen::Matrix<double, 3, 1> &obs,
                    const uint32_t &t, uint32_t &index, Parameters params)
    {
        uint32_t obs_thick = obs(1, 0);
        uint32_t obs_thick_d2 = obs_thick / 2;

        uint32_t obs_n_min = obs(0, 0) - obs_thick_d2;
        if (obs_n_min != 0)
            obs_n_min--;

        uint32_t obs_n_max = obs(0, 0) + obs_thick_d2 + 1;

        // Only checking the acceptation for near predictions
        //    while (index < filters.size() && filters[index].X_predicted(0, 0) - obs_n_max < 10)
        index = 0;
        while (index < filters.size())
        {
            Filter &f = filters[index];
            if (accepts(f, obs, obs_n_min, obs_n_max, params))
                accepted.push_back(&f);
            else if (f.observation == std::nullopt && in_between(obs_n_min, f.n_min, obs_n_max) &&
                     in_between(obs_n_min, f.n_max, obs_n_max) && f.X_predicted(1, 0) < obs_thick)
            {
                add_point_under_other(f, t, round(f.X_predicted(0, 0)), round(f.X_predicted(1, 0)));
            }
            index++;
        }

        // Check the overflow
        if (index == filters.size() && !filters.empty())
            index--;

        // Go back with index
        while (index > 0 && filters[index].X_predicted(0, 0) - obs_n_min > 0)
            index--;
    }

    /**
   * Check if the filters are following the same observation
   * @param f first filter
   * @param fj second filter
   * @return true if the filters are following the same observation for SEGDET_MINIMUM_FOR_FUSION length
   */
    bool same_observation(const Filter &f, const Filter &fj, Parameters params)
    {
        if (f.n_values.size() < params.minimum_for_fusion || fj.n_values.size() < params.minimum_for_fusion)
            return false;

        for (uint32_t i = 0; i < params.minimum_for_fusion; i++)
        {
            size_t k = f.thicknesses.size() - 1 - i;
            size_t kj = fj.thicknesses.size() - 1 - i;

            if (f.thicknesses[k] != fj.thicknesses[kj] || f.t_values[k] != fj.t_values[kj] ||
                f.n_values[k] != fj.n_values[kj])
                return false;
        }

        return true;
    }

    /**
   * Extract the segment from a filter
   * @param filter filter from which we create the segment
   * @param min_len
   * @param nb_to_remove number of last pixels to remove
   * @return the segment created calling the constructor
   */
    Segment make_segment_from_filter(Filter &filter, uint32_t min_len, uint32_t nb_to_remove)
    {
        uint32_t last_index = filter.n_values.size() - nb_to_remove;

        for (uint32_t i = 0; i < last_index; i++)
        {
            filter.segment_points.emplace_back(filter.n_values[i], filter.t_values[i], filter.thicknesses[i],
                                               filter.is_horizontal);
        }
        auto &point = filter.segment_points;
        auto &under_other = filter.under_other;

        for (auto &p : filter.currently_under_other)
            under_other.push_back(p);

        auto first_part_slope =
            std::nullopt == filter.first_slope ? filter.slopes[min_len - 1] : filter.first_slope.value();
        auto last_part_slope = filter.slopes[last_index - 1];

        return Segment(point, under_other, first_part_slope, last_part_slope, filter.is_horizontal);
    }

    /**
   * Erase a filter adding if needed the extract segment
   * @param filters Current list of filters
   * @param segments Current list of segments
   * @param min_len
   * @param j Index of the filter to erase
   * @param fj Filter to erase
   */
    void erase_filter(std::vector<Filter> &filters, std::vector<Segment> &segments, uint32_t min_len, uint32_t j,
                      Filter &fj, Parameters params)
    {
        if (fj.last_integration - fj.first - params.minimum_for_fusion > min_len)
            segments.push_back(make_segment_from_filter(fj, min_len, params.minimum_for_fusion));

        filters.erase(filters.begin() + j);
    }

    /**
   * Check if a fusion with the current filter
   * @param filters Current list of filter
   * @param index Index of the current filter
   * @param segments Current list of segment
   * @param min_len
   * @return true if happened
   */
    bool make_potential_fusion(std::vector<Filter> &filters, uint32_t index, std::vector<Segment> &segments,
                               uint32_t min_len, Parameters params)
    {
        auto f = filters[index];

        uint32_t j = index + 1;
        bool current_filter_was_deleted = false;

        while (j < filters.size())
        {
            auto fj = filters[j];
            if (fj.observation != std::nullopt && same_observation(f, fj, params))
            {
                if (f.first < fj.first)
                    erase_filter(filters, segments, min_len, j, fj, params);
                else
                {
                    current_filter_was_deleted = true;
                    erase_filter(filters, segments, min_len, index, f, params);
                    break;
                }
            }

            j++;
        }

        return current_filter_was_deleted;
    }

    /**
   * Say if the filter has to continue
   * @param f filter to check
   * @param t current t
   * @param discontinuity discontinuity allowedt
   * @return true if the filter has to continue
   */
    bool filter_has_to_continue(Filter &f, uint32_t t, uint32_t discontinuity)
    {
        if (t - f.last_integration <= discontinuity)
            return true;

        if (!f.currently_under_other.empty())
        {
            auto last_t = f.is_horizontal ? f.currently_under_other[f.currently_under_other.size() - 1].x
                                          : f.currently_under_other[f.currently_under_other.size() - 1].y;
            return t - last_t <= discontinuity;
        }

        return false;
    }

    /**
   * Integrate observation for filters that matched an observation
   * Select filters that have to continue
   * Add segment in segments if needed
   * @param filters Current filters
   * @param segments Current segments
   * @param t Current t
   * @param two_matches Current time of two filters matching
   * @param min_len_embryo
   * @param discontinuity
   * @return List of filters that have to continue
   */
    std::vector<Filter> filter_selection(std::vector<Filter> &filters, std::vector<Segment> &segments, uint32_t t,
                                         uint32_t two_matches, uint32_t min_len_embryo, uint32_t discontinuity,
                                         Parameters params)
    {
        std::vector<Filter> filters_to_keep;

        size_t index = 0;
        while (index < filters.size())
        {
            auto f = filters[index];

            if (f.observation != std::nullopt)
            {
                if (two_matches > params.minimum_for_fusion &&
                    make_potential_fusion(filters, index, segments, min_len_embryo, params))
                    continue;

                integrate(f, t, params);
                compute_sigmas(f, params);

                if (f.nb_current_slopes_over_slope_max > params.max_slopes_too_large)
                {
                    if (f.last_integration - f.first - params.max_slopes_too_large > min_len_embryo)
                        segments.push_back(make_segment_from_filter(f, min_len_embryo, 0));
                    index++;
                    continue;
                }

                filters_to_keep.push_back(f);
            }
            else if (filter_has_to_continue(f, t, discontinuity))
            {
                f.S = f.S_predicted;
                filters_to_keep.push_back(f);
            }
            else if (f.last_integration - f.first > min_len_embryo)
                segments.push_back(make_segment_from_filter(f, min_len_embryo, 0));

            index++;
        }

        return filters_to_keep;
    }

    /**
   * Add last segment from current filters if needed
   * @param filters Current filters
   * @param segments Current segments
   * @param min_len
   */
    void to_thrash(std::vector<Filter> &filters, std::vector<Segment> &segments, uint32_t min_len, Parameters params)
    {
        uint32_t i = 0;
        while (i < filters.size())
        {
            auto &f = filters[i];
            if (make_potential_fusion(filters, i, segments, min_len, params))
                continue;

            if (f.last_integration - f.first > min_len)
                segments.push_back(make_segment_from_filter(f, min_len, 0));

            i++;
        }
    }

    /**
   * Set the parameters of the traversal according to the direction
   * @param is_horizontal direction
   * @param xmult
   * @param ymult
   * @param slope_max
   * @param n_max
   * @param t_max
   * @param width Width of the image
   * @param height Height of the image
   */
    void set_parameters(bool is_horizontal, uint32_t &xmult, uint32_t &ymult, double &slope_max, uint32_t &n_max,
                        uint32_t &t_max, uint32_t width, uint32_t height, Parameters params)
    {
        if (is_horizontal)
        {
            xmult = 0;
            ymult = 1;
            slope_max = params.slope_max_horizontal;
            n_max = height;
            t_max = width;
        }
        else
        {
            xmult = 1;
            ymult = 0;
            slope_max = params.slope_max_vertical;
            n_max = width;
            t_max = height;
        }
    }

    /**
   * Handle case where one or more filter matched the observation
   * @param new_filters List of new filters
   * @param accepted List of filters that accepts the observation
   * @param obs
   * @param t
   * @param is_horizontal
   * @param slope_max
   * @return
   */
    bool handle_find_filter(std::vector<Filter> &new_filters, std::vector<Filter *> &accepted,
                            const Eigen::Matrix<double, 3, 1> &obs, uint32_t &t, bool is_horizontal, double slope_max)
    {
        auto observation_s = Observation(obs, accepted.size(), t);

        for (auto &f : accepted)
        {
            auto obs_result = choose_nearest(*f, observation_s);

            if (obs_result != std::nullopt)
            {
                auto obs_result_value = obs_result.value();
                obs_result_value.match_count--;

                if (obs_result_value.match_count == 0)
                    new_filters.emplace_back(is_horizontal, t, slope_max, obs_result_value.obs);
            }
        }

        return observation_s.match_count > 1;
    }

    /**
   * Merge selection and new_filters in filters
   * @param filters
   * @param selection
   * @param new_filters
   */
    void update_current_filters(std::vector<Filter> &filters, std::vector<Filter> &selection,
                                std::vector<Filter> &new_filters)
    {
        filters.clear();

        for (auto &f : selection)
            filters.push_back(f);

        for (auto &f : new_filters)
            filters.push_back(f);
    }

    std::vector<Segment> traversal_batch(const image2d<uint8_t> &image, bool is_horizontal, uint min_len_embryo,
                                   uint discontinuity, Parameters params)
    {
        // Usefull parameter used in the function
        uint32_t xmult, ymult;
        double slope_max;
        uint32_t n_max, t_max;

        set_parameters(is_horizontal, xmult, ymult, slope_max, n_max, t_max, image.size(), image.size(1), params);

        auto filters = std::vector<Filter>();   // List of current filters
        auto segments = std::vector<Segment>(); // List of current segments
        std::vector<Filter> new_filters{};
        std::vector<Filter> selection{};

        uint32_t two_matches = 0; // Number of t where two segments matched the same observation
        // Useful to NOT check if filters has to be merged

        std::vector<std::vector<std::pair<int, int>>> observations;

        auto p = obs_parser();
        if (!is_horizontal)
        {
            auto tr_image = image.copy();
            tr_image.transpose();
            observations = p.parse(tr_image.width, tr_image.height, tr_image.get_buffer_const(), params.max_llum);
        }
        else
            observations = p.parse(image.width, image.height, image.get_buffer_const(), params.max_llum);
        
        for (uint32_t t = 0; t < t_max; t++)
        {
            for (auto &filter : filters)
                predict(filter);

            new_filters.clear();
            bool two_matches_through_n = false;
            uint32_t filter_index = 0;

        
            for (auto& vec_obs : observations[t])
            {
                Eigen::Matrix<double, 3, 1> obs = Eigen::Vector3d({(double) vec_obs.first, (double) vec_obs.second, (double) 200.0});

                std::vector<Filter *> accepted{}; // List of accepted filters by the current observation obs
                find_match(filters, accepted, obs, t, filter_index, params);
                if (accepted.empty() && obs(1, 0) < params.max_thickness)
                    new_filters.emplace_back(is_horizontal, t, slope_max, obs);
                else
                    two_matches_through_n =
                        handle_find_filter(new_filters, accepted, obs, t, is_horizontal, slope_max) || two_matches_through_n;
            }
            

            if (two_matches_through_n)
                two_matches++;
            else
                two_matches = 0;

            // Selection for next turn
            selection = filter_selection(filters, segments, t, two_matches, min_len_embryo, discontinuity, params);
            // Merge selection and new_filters in filters
            update_current_filters(filters, selection, new_filters);
        }

        to_thrash(filters, segments, min_len_embryo, params);

        return segments;
    }

    std::vector<Segment> traversal(const image2d<uint8_t> &image, bool is_horizontal, uint min_len_embryo,
                                   uint discontinuity, Parameters params)
    {
        // Usefull parameter used in the function
        uint32_t xmult, ymult;
        double slope_max;
        uint32_t n_max, t_max;

        set_parameters(is_horizontal, xmult, ymult, slope_max, n_max, t_max, image.size(), image.size(1), params);

        auto filters = std::vector<Filter>();   // List of current filters
        auto segments = std::vector<Segment>(); // List of current segments
        std::vector<Filter> new_filters{};
        std::vector<Filter> selection{};

        uint32_t two_matches = 0; // Number of t where two segments matched the same observation
        // Useful to NOT check if filters has to be merged

        std::vector<std::vector<std::pair<int, int>>> observations;

        for (uint32_t t = 0; t < t_max; t++)
        {
            for (auto &filter : filters)
                predict(filter);

            new_filters.clear();
            bool two_matches_through_n = false;
            uint32_t filter_index = 0;


            for (uint32_t n = 0; n < n_max; n++)
            {
                if (image_at(image, n, t, is_horizontal) < params.max_llum)
                {
                    Eigen::Matrix<double, 3, 1> obs = determine_observation(image, n, t, n_max, is_horizontal, params);

                    std::vector<Filter *> accepted{}; // List of accepted filters by the current observation obs
                    find_match(filters, accepted, obs, t, filter_index, params);
                    if (accepted.empty() && obs(1, 0) < params.max_thickness)
                        new_filters.emplace_back(is_horizontal, t, slope_max, obs);
                    else
                        two_matches_through_n =
                            handle_find_filter(new_filters, accepted, obs, t, is_horizontal, slope_max) || two_matches_through_n;
                }
            }
            
            if (two_matches_through_n)
                two_matches++;
            else
                two_matches = 0;

            // Selection for next turn
            selection = filter_selection(filters, segments, t, two_matches, min_len_embryo, discontinuity, params);
            // Merge selection and new_filters in filters
            update_current_filters(filters, selection, new_filters);
        }

        to_thrash(filters, segments, min_len_embryo, params);

        return segments;
    }

    /**
   * Compute the distance between 2 points
   * @param p1
   * @param p2
   * @return
   */
    double distance_points(const Point &p1, const Point &p2)
    {
        int xvar = (int)p1.x - (int)p2.x;
        int yvar = (int)p1.y - (int)p2.y;

        xvar *= xvar;
        yvar *= yvar;

        return std::sqrt(xvar + yvar);
    }

    /**
   * Say if 2 segments have to be linked
   * @param a
   * @param b
   * @return
   */
    double distance_linking(Segment &a, const Segment &b, const Parameters &params)
    {
        if (std::abs(a.last_part_slope - b.first_part_slope) > params.merge_slope_variation)
            return params.merge_distance_max;

        return distance_points(a.last_point, b.first_point);
    }

    /**
   * Merge the second segment to the first
   * @param a First segment
   * @param b Second segment
   */
    void merge_segments(Segment &a, const Segment &b)
    {
        if (a.is_horizontal)
            a.length = b.points[b.points.size() - 1].x - a.points[0].x + 1;
        else
            a.length = b.points[b.points.size() - 1].y - a.points[0].y + 1;

        for (auto &p : b.points)
            a.points.push_back(p);

        a.nb_pixels += b.nb_pixels;

        a.last_part_slope = b.last_part_slope;
        a.last_point = b.last_point;
    }

    /**
   * Call the function linking segments if needed
   * @param segments All segments
   */
    void segment_link(std::vector<Segment> &segments, Parameters params)
    {
        size_t i = 0;
        while (i < segments.size())
        {
            size_t j = i + 1;

            size_t best_index = i;
            double best_distance = params.merge_distance_max;
            while (j < segments.size())
            {
                auto distance_link = distance_linking(segments[i], segments[j], params);
                if (distance_link < best_distance)
                {
                    best_index = j;
                    best_distance = distance_link;
                }
                else
                    j++;
            }

            if (best_distance < params.merge_distance_max)
            {
                merge_segments(segments[i], segments[best_index]);
                segments.erase(segments.begin() + best_index);
            }

            i++;
        }
    }

    /**
   * Call segment link for horizontal and vertical segments
   * @param pair First is horizontal segment and second vertical segments
   */
    void segment_linking(std::pair<std::vector<Segment>, std::vector<Segment>> &pair, Parameters params)
    {
        segment_link(pair.first, params);
        segment_link(pair.second, params);
    }

    /**
   * Label a pixel in img
   * @param img Image to labelize
   * @param label Color
   * @param x X position
   * @param y Y position
   */
    void draw_labeled_pixel(image2d<uint16_t> &img, uint16_t label, int x, int y)
    {
        img({x, y}) = img({x, y}) != 0 ? 1 : label;
    }

    /**
   * Draw a pint in img
   * @param img Image to labelize
   * @param label Color
   * @param point Point to draw
   * @param is_horizontal
   */
    void draw_labeled_point(image2d<uint16_t> &img, uint16_t label, Point point, bool is_horizontal)
    {
        auto thickness = point.thickness / 2;
        auto is_odd = point.thickness % 2;

        if (is_horizontal)
        {
            for (int i = -thickness; i < static_cast<int>(thickness) + static_cast<int>(is_odd); i++)
            {
                if (static_cast<int>(point.y) + i < img.size(1))
                    draw_labeled_pixel(img, label, static_cast<int>(point.x), static_cast<int>(point.y + i));
            }
        }
        else
        {
            for (int i = -thickness; i < static_cast<int>(thickness) + static_cast<int>(is_odd); i++)
            {
                if (static_cast<int>(point.x) + i < img.size(0))
                    draw_labeled_pixel(img, label, static_cast<int>(point.x + i), static_cast<int>(point.y));
            }
        }
    }

    enum labeling_type
    {
        LABELING_TYPE_VERTICAL,
        LABELING_TYPE_HORIZONTAL,
    };

    void labeled_arr(image2d<uint16_t> &out, const std::vector<Segment> &segments)
    {
        for (size_t i = 0; i < segments.size(); i++)
        {
            for (auto &point : segments[i].points)
                draw_labeled_point(out, i + 3, point, segments[i].is_horizontal);
        }
    }

    /**
   * Draw segments in img out according to type
   * @param out Image to draw in
   * @param horizontal_segments
   * @param vertical_segments
   * @param type Labeling type
   */
    void labeled_arr(image2d<uint16_t> out, std::vector<Segment> &horizontal_segments,
                     std::vector<Segment> &vertical_segments, labeling_type type)
    {
        std::vector<Segment> segments = type == LABELING_TYPE_HORIZONTAL ? horizontal_segments : vertical_segments;

        labeled_arr(out, segments);
    }

    /**
   * Binarize the image
   * @param img Image to binarize
   */
    void binarize_img(image2d<uint16_t> img)
    {
        for (auto &p : img.domain())
        {
            if (img(p) != 0)
                img(p) = 1;
            else
                img(p) = 0;
        }
        // mln_foreach(auto p, img.domain())
        // {
        //     if (img(p) != 0)
        //         img(p) = 1;
        //     else
        //         img(p) = 0;
        // }
    }

    /**
   * Compute the intersection between two images and store it in out
   * @param img first image
   * @param img2 second image
   * @param out image storing result
   */
    void intersect(image2d<uint16_t> img, image2d<uint16_t> img2, image2d<uint16_t> out)
    {
        for (auto &pt : img.domain())
            out(pt) = img(pt) * img2(pt);
        // mln_foreach(auto pt, img.domain())
        //     out(pt) = img(pt) * img2(pt);
    }

    /**
   * Remove duplication of segment
   * @param segments_to_compare
   * @param segments_removable
   * @param width
   * @param height
   */
    void remove_dup(std::vector<Segment> &segments_to_compare, std::vector<Segment> &segments_removable, size_t width,
                    size_t height, Parameters params)
    {
        image2d<uint16_t> first_output = image2d<uint16_t>(width, height);
        // mln::fill(first_output, 0);
        first_output.fill(0);
        labeled_arr(first_output, segments_to_compare, segments_removable, LABELING_TYPE_HORIZONTAL);

        image2d<uint16_t> second_output = image2d<uint16_t>(width, height);
        second_output.fill(0);
        // mln::fill(second_output, 0);
        labeled_arr(second_output, segments_to_compare, segments_removable, LABELING_TYPE_VERTICAL);

        // auto second_output_bin =
        //     mln::view::transform(second_output, [](uint16_t p) -> uint8_t
        //                          { return (p != 0) ? 1 : 0; });

        auto second_output_bin = second_output.copy().transform([](uint16_t p) -> uint16_t
                                  { return (p != 0) ? 1 : 0; });

        binarize_img(first_output);

        image2d<uint16_t> intersection = image2d<uint16_t>(width, height);
        intersection.fill(0);

        intersect(first_output, second_output, intersection);

        std::vector<uint16_t> segments = std::vector<uint16_t>(segments_removable.size());

        for (unsigned short &segment : segments)
            segment = 0;

        // mln_foreach(auto v, intersection.values())
        // {
        //     if (v >= 3)
        //         segments[v - 3]++;
        // }

        for (auto& p : intersection.domain())
        {
            auto v = intersection(p); 
            if (v >= 3)
                segments[v - 3]++;
        }

        int k = 0;
        for (size_t i = 0; i < segments.size(); i++)
        {
            double segments_ratio = 0;
            if (segments_removable[i - k].nb_pixels != 0)
                segments_ratio = segments[i] / (double)segments_removable[i - k].nb_pixels;
            if (segments_removable[i - k].nb_pixels == 0 || segments_ratio > params.threshold_intersection)
                segments_removable.erase(segments_removable.begin() + i - k);
            k++;
        }
    }

    /**
   * Call remove duplicates for horizontal and vertical segments
   * @param pair Pair (horizontal segments,vertical segments)
   * @param img_width Width of the image where segments were extract
   * @param img_height Height of the image where segments were extract
   */
    void remove_duplicates(std::pair<std::vector<Segment>, std::vector<Segment>> &pair, size_t img_width,
                           size_t img_height, Parameters params)
    {
        remove_dup(pair.first, pair.second, img_width, img_height, params);
        remove_dup(pair.second, pair.first, img_width, img_height, params);
    }

    /**
   * Post process segments linking them and removing duplications
   * @param pair Pair (horizontal segments,vertical segments)
   * @param img_width Width of the image where segments were extract
   * @param img_height Height of the image where segments were extract
   */
    void post_process(std::pair<std::vector<Segment>, std::vector<Segment>> &pair, size_t img_width, size_t img_height,
                      Parameters params)
    {
        segment_linking(pair, params);
        remove_duplicates(pair, img_width, img_height, params);
    }

    /**
   * Compute the two traversals to detect horizontal and vertical segments
   * @param image image to extract segment from
   * @param min_len
   * @param discontinuity
   * @return Pair (horizontal segments,vertical segments)
   */
    std::pair<std::vector<Segment>, std::vector<Segment>> process(const image2d<uint8_t> &image, uint min_len_embryo,
                                                                  uint discontinuity, Parameters params, const std::string& mode)
    {
        // TODO Multi threading, splitter l'image
        std::vector<Segment> horizontal_segments;
        std::vector<Segment> vertical_segments;

        if (mode == "batch")
        {
            horizontal_segments = traversal_batch(image, true, min_len_embryo, discontinuity, params);
            vertical_segments = traversal_batch(image, false, min_len_embryo, discontinuity, params);
        }
        else
        {
            horizontal_segments = traversal(image, true, min_len_embryo, discontinuity, params);
            vertical_segments = traversal(image, false, min_len_embryo, discontinuity, params);
        }


        return std::make_pair(horizontal_segments, vertical_segments);
    }

    std::vector<Segment> filter_length(std::pair<std::vector<Segment>, std::vector<Segment>> &p, uint min_len)
    {
        std::vector<Segment> res{};

        for (auto &seg : p.first)
        {
            if (seg.length > min_len)
                res.push_back(seg);
        }
        for (auto &seg : p.second)
        {
            if (seg.length > min_len)
                res.push_back(seg);
        }

        return res;
    }

    // Public functions


    std::vector<Segment> detect_line(image2d<uint8_t>& image, int min_len, int discontinuity, const Parameters &params, const std::string& mode)
    {
        // Preprocessing not done because its mathematical morphology, and we do not want to parallellize it
        // mln::ndbuffer_image preprocessed_img = preprocess(std::move(image));

        uint min_len_embryo = min_len / 4 + 1; // 5U < min_len ? 5U : min_len;
        // Processing
        auto p = process(image, min_len_embryo, discontinuity, params, mode);

        // Post Processing
        post_process(p, image.size(0), image.size(1), params);

        auto res = filter_length(p, min_len);

        return res;
    }

    std::vector<Segment> detect_line(image2d<uint8_t>& image, int min_len, int discontinuity, const std::string& mode)
    {
        return detect_line(image, min_len, discontinuity, Parameters(), mode);
    }
} // namespace mln::contrib::segdet