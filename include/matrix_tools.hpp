#pragma once

#include <vector>
#include <iostream>

namespace kalman
{
 
template<typename T>
class kMatrix
{
public:
    std::vector<T> buffer;
    int nb_rows;
    int nb_cols;

    kMatrix()
    : buffer(std::vector<float>(3)),
    nb_rows(3),
    nb_cols(1)
    {
    }

    kMatrix(std::vector<T> buffer, int nb_rows, int nb_cols)
            : buffer(buffer), nb_rows(nb_rows),
              nb_cols(nb_cols)
    {}
    T& operator()(int i, int j)
    {
        return buffer[i + j * nb_rows];
    }
    T operator()(int i, int j) const
    {
        return buffer[i + j * nb_rows];
    }
};

std::vector<float> invert_matrix(const std::vector<float> &mat, size_t len);

float compute_det(const std::vector<float> &mat, size_t len);

std::vector<float> matmul(const std::vector<float> &lhs, const std::vector<float> &rhs, size_t n, size_t p,
                           size_t m);

template <typename T>
kMatrix<T> operator*(const kMatrix<T> &lhs, const kMatrix<T> &rhs)
{
    return matmul(lhs, rhs);
}

template <typename T>
kMatrix<T> operator+(const kMatrix<T> &lhs, const kMatrix<T> &rhs)
{
    if (lhs.nb_rows != rhs.nb_rows || lhs.nb_cols != rhs.nb_cols)
    {
        throw std::runtime_error("Can't add matrices with different dimensions");
    }
    auto out_buffer = std::vector<float>(lhs.nb_cols * lhs.nb_rows);
    for (int i = 0; i < lhs.nb_cols * lhs.nb_rows; i++)
    {
        out_buffer[i] = lhs.buffer[i] + rhs.buffer[i];
    }
    return kMatrix<float>(out_buffer, lhs.nb_rows, lhs.nb_cols);
}

template <typename T>
kMatrix<T> operator-(const kMatrix<T> &lhs, const kMatrix<T> &rhs)
{
    if (lhs.nb_rows != rhs.nb_rows || lhs.nb_cols != rhs.nb_cols)
    {
        throw std::runtime_error("Can't add matrices with different dimensions");
    }
    auto out_buffer = std::vector<float>(lhs.nb_cols * lhs.nb_rows);
    for (int i = 0; i < lhs.nb_cols * lhs.nb_rows; i++)
    {
        out_buffer[i] = lhs.buffer[i] - rhs.buffer[i];
    }
    return kMatrix<float>(out_buffer, lhs.nb_rows, lhs.nb_cols);
}

template <typename T>
kMatrix<T> invert_matrix3(const kMatrix<T> &mat)
{
    auto inv_buf = invert_matrix(mat.buffer, 3);
    return kMatrix<float>(inv_buf, 3, 3);
}

template <typename T>
kMatrix<T> matmul(const kMatrix<T> &lhs, const kMatrix<T> &rhs)
{
    if (lhs.nb_cols != rhs.nb_rows)
    {
        throw std::runtime_error("Matmul : dimensions do not match");
    }
    auto &m1 = lhs.buffer;
    auto &m2 = rhs.buffer;
    auto mult = matmul(m1, m2, lhs.nb_rows, lhs.nb_cols, rhs.nb_cols);

    return kMatrix<T>(mult, lhs.nb_rows, rhs.nb_cols);
}


std::vector<float> get_adjugate_matrix(const std::vector<float> &mat, size_t len);
   
}