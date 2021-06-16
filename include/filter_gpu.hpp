#pragma once

#include "segment.hpp"
#include "matrix_tools.hpp"
#include <utility>
#include <optional>
#include <vector>

#define SEGDET_NB_VALUES_TO_KEEP 30

#define SEGDET_VARIANCE_POSITION 2
#define SEGDET_VARIANCE_THICKNESS 1
#define SEGDET_VARIANCE_LUMINOSITY 12

#define SEGDET_DEFAULT_SIGMA_POS 2
#define SEGDET_DEFAULT_SIGMA_THK 2
#define SEGDET_DEFAULT_SIGMA_LUM 57

namespace kalman_gpu
{
    using namespace kalman;
  /**
   * The Observation class holds an observation matrix, the t-position at which it was matched, and the number of times
   * this observation was matched.
   */
  struct Observation
  {
    /**
     * Constructor
     * @param obs The observation matrix
     * @param matchCount The number of time this observation was matched
     * @param t The position at which the observation was made
     */
    Observation(kMatrix<float> obs, uint32_t matchCount, uint32_t t)
      : obs(std::move(obs))
      , match_count(matchCount)
      , t(t)
    {
    }
//    Eigen::Matrix<float, 3, 1> obs;         // The observation Matrix
    kMatrix<float> obs;         // The observation Matrix
    uint32_t                    match_count; // The numbers of time the observation was matched to a filter
    uint32_t                    t;           // The t-position at which the observation was made
  };

  static const kMatrix<float> A({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4);
  static const kMatrix<float> A_transpose({1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4);
  static const kMatrix<float> C({1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 3, 4);
  static const kMatrix<float> C_transpose({1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1}, 4, 3);
  static const kMatrix<float> Vn({SEGDET_VARIANCE_POSITION, 0, 0, 0,
                                   SEGDET_VARIANCE_THICKNESS, 0, 0, 0, SEGDET_VARIANCE_LUMINOSITY}, 3, 3);

  /**
   * The Filter struct represents an ongoing kalman filter
   */
  struct Filter
  {
    /**
     * Constructor
     * @param isHorizontal Boolean set to true if the filter is horizontal
     * @param t_integration The position of the first integration
     * @param slopeMax The maximum value that the filter's slope can have
     * @param observation The filter's first observation
     */
//    Filter(bool isHorizontal, uint32_t t_integration, float slopeMax, Eigen::Matrix<float, 3, 1> observation)
      Filter(bool isHorizontal, uint32_t t_integration, float slopeMax, kMatrix<float> observation)
      : is_horizontal(isHorizontal)
      , slope_max(slopeMax)
      , S(kMatrix<float>({observation(0, 0), 0, observation(1, 0), observation(2, 0)}, 4, 1))
      , W(kMatrix<float>({0, 0, 0, 0}, 4, 1))
//      , W(Eigen::Matrix<float, 4, 1>::Zero())
      , N(kMatrix<float>({0,0,0}, 3, 1))
//      , N(Eigen::Matrix<float, 3, 1>::Zero())
      , H(kMatrix<float>({1, 0, 0, 0,
                               0, 1, 0, 0,
                               0, 0, 1, 0 ,
                               0, 0, 0,1}, 4, 4))
//      , H(Eigen::Matrix<float, 4, 4>::Identity())
      , observation(std::nullopt)
      , observation_distance(0)
      , last_integration(t_integration)
      , S_predicted(kMatrix<float>({0, 0,0, 0}, 4, 1))
      , X_predicted(kMatrix<float>({0, 0, 0}, 3, 1))
    {
      first    = t_integration;
      t_values = std::vector<uint32_t>({t_integration});
      n_values = std::vector<uint32_t>({static_cast<uint32_t>(observation(0, 0))});

      slopes                           = std::vector<float>({0});
      thicknesses                      = std::vector<float>({observation(1, 0)});
      luminosities                     = std::vector<float>({observation(2, 0)});
      nb_current_slopes_over_slope_max = 0;
      first_slope                      = std::nullopt;

      under_other           = std::vector<Point>();
      currently_under_other = std::vector<Point>();
      segment_points        = std::vector<Point>();

      n_min = 0;
      n_max = 0;

      sigma_position   = SEGDET_DEFAULT_SIGMA_POS;
      sigma_thickness  = SEGDET_DEFAULT_SIGMA_THK;
      sigma_luminosity = SEGDET_DEFAULT_SIGMA_LUM;

//      S_predicted = Eigen::Matrix<float, 4, 1>::Zero();
//      X_predicted = Eigen::Matrix<float, 3, 1>::Zero();

    }

    bool      is_horizontal; // set if filter is horizontal (t = x, n = y)
    u_int32_t first;         // t value of first integration

    std::vector<uint32_t> t_values; // t values of SEGDET_NB_VALUES_TO_KEEP last integrations
    std::vector<uint32_t> n_values;

    uint                  nb_current_slopes_over_slope_max;
    float                slope_max;
    std::optional<float> first_slope;
    std::vector<float>   thicknesses;
    std::vector<float>   luminosities;
    std::vector<float>   slopes;

    std::vector<Point> under_other; //
    std::vector<Point> currently_under_other;
    std::vector<Point> segment_points;

//    Eigen::Matrix<float, 4, 1> S; // state matrix {{position (n)}, {slope}, {thickness}, {luminosity}}
//    Eigen::Matrix<float, 4, 1> W; // noise matrix
//    Eigen::Matrix<float, 3, 1> N; // measured noise matrix
//    Eigen::Matrix<float, 4, 4> H; // S prediction error variance matrix
    kMatrix<float> S; // state matrix {{position (n)}, {slope}, {thickness}, {luminosity}}
    kMatrix<float> W; // noise matrix
    kMatrix<float> N; // measured noise matrix
    kMatrix<float> H; // S prediction error variance matrix

    float n_min;
    float n_max;
    float sigma_position;
    float sigma_thickness;
    float sigma_luminosity;

//    Eigen::Matrix<float, 4, 1> S_predicted;
//    Eigen::Matrix<float, 3, 1> X_predicted;
    kMatrix<float> S_predicted;
    kMatrix<float> X_predicted;

    int observation_index;
    std::optional<Observation>
              observation;          // matrix {{position (n)},{thickness},{luminosity}}, nullopt if none was matched
    u_int32_t observation_distance; // n distance from last observation to current prediction

    u_int32_t last_integration; // t value referring to the position of the last integration
  };

  /**
   * The predict method, given a filter f, will update its predicted state
   * @param f The filter for which to update
   */
  void predict(Filter& f);

  /**
   * Update the given filter if enough values by computing the standard deviations of the position, thickness and
   * luminosity vectors
   * @param f
   */
  void compute_sigmas(Filter& f, Parameters params);

  /**
   * The accepts method will check if the given filter and observation are compatible
   * @param filter The filter struct
   * @param obs The observation Matrix
   * @param min The observation min value
   * @param max The observation max value
   * @return true if observation is compatible with filter else false
   */
  bool accepts(const Filter& filter, const kMatrix<float>& obs, uint32_t min, uint32_t max,
               Parameters params);

  /**
   * The choose_nearest method check whether the given observation, or the one already contained in the Filter is the
   * most compatible
   * @param f The filter struct
   * @param obs The observation to check
   * @return The best observation for the filter
   */
  std::optional<Observation> choose_nearest(Filter& f, Observation& obs, int obs_index);

  /**
   * The integrate method, given a filter f, will update the state with the a new value
   * @param f The filter struct
   * @param t The position of the new last integration
   */
  void integrate(Filter& f, uint32_t t, Parameters params);


} // namespace mln::contrib::segdet
