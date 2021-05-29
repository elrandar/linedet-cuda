#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace mln::contrib::segdet
{
  template <typename T>
  double mean(std::vector<T> array, size_t start, size_t end)
  {
    auto it_end = end == 0 ? array.end() : array.begin() + end;
    return std::accumulate(array.begin() + start, it_end, 0.0) / static_cast<double>(array.size());
  }

  template <typename X, typename Z>
  class Linear_Regression
  {

  public:
    template <typename T>
    std::vector<T> vec_multiply(std::vector<T> array, std::vector<T> array2)
    {
      std::vector<T> out(array2.size());
      for (size_t i = 0; i < array.size(); i++)
      {
        out[i] = array[i] * array2[i];
      }
      return out;
    }


    template <typename T, typename M>
    void estimate_coef(std::vector<T> indep_var, std::vector<T> dep_var, M& B_1, M& B_0)
    {
      M   X_mean = mean(indep_var, 0, 0);
      M   Y_mean = mean(dep_var, 0, 0);
      M   SS_xy  = 0;
      M   SS_xx  = 0;
      int n      = indep_var.size();
      {
        std::vector<T> temp;
        temp  = vec_multiply(indep_var, dep_var);
        SS_xy = std::accumulate(temp.begin(), temp.end(), 0);
        SS_xy = SS_xy - n * X_mean * Y_mean;
      }
      {
        std::vector<T> temp;
        temp  = vec_multiply(indep_var, indep_var);
        SS_xx = std::accumulate(temp.begin(), temp.end(), 0);
        SS_xx = SS_xx - n * X_mean * X_mean;
      }

      B_1 = SS_xy / SS_xx;
      B_0 = Y_mean - B_1 * X_mean;
    }

    void fit(std::vector<std::vector<Z>> dataset) { estimate_coef(dataset[0], dataset[1], b_1, b_0); }

    Z predict(const Z& test_data) { return b_0 + (b_1 * test_data); }
    X b_1;
    X b_0;
  };
} // namespace mln::contrib::segdet
