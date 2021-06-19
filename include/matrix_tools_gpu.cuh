#pragma once

#include "cuda_header.hpp"
#include <stdlib.h>


namespace kalman_gpu
{


template<typename T, int H, int W>
class kMatrix
{
public:
    T buffer[H * W];
    int nb_rows = H;
    int nb_cols = W;

    CUDA_CALLABLE_MEMBER kMatrix()
    {
        for (int i = 0; i < H * W; i++)
            buffer[i] = 0;
    }
    CUDA_CALLABLE_MEMBER kMatrix(T* buf)
    {
        for (int i = 0; i < H * W; i++)
            buffer[i] = buf[i];
    }
    CUDA_CALLABLE_MEMBER T& operator()(int i, int j)
    {
        return buffer[i + j * nb_rows];
    }
    CUDA_CALLABLE_MEMBER T operator()(int i, int j) const
    {
        return buffer[i + j * nb_rows];
    }
};


CUDA_CALLABLE_MEMBER float* invert_matrix(float* mat, size_t len);

CUDA_CALLABLE_MEMBER float compute_det(float* mat, size_t len);

CUDA_CALLABLE_MEMBER float* matmul(float* lhs, float* rhs, float* out, size_t n, size_t p,
                           size_t m);

CUDA_CALLABLE_MEMBER float* matmul(const float* lhs, const float* rhs, float* out, const size_t n, const size_t p,
                           const size_t m);

                           
template <typename T, int H, int W>
CUDA_CALLABLE_MEMBER kMatrix<T, H, W> operator*(const kMatrix<T, H, W> &lhs, const kMatrix<T, H, W> &rhs)
{
    return matmul(lhs, rhs);
}

template <typename T, int H, int W>
CUDA_CALLABLE_MEMBER kMatrix<T, H, W> add(const kMatrix<T, H, W> &lhs, const kMatrix<T, H, W> &rhs,
                                          kMatrix<T, H, W> &out)
{
    T* out_buffer = out.buffer;
    for (int i = 0; i < lhs.nb_cols * lhs.nb_rows; i++)
    {
        out_buffer[i] = lhs.buffer[i] + rhs.buffer[i];
    }
    return out;
}

template <typename T, int H, int W>
CUDA_CALLABLE_MEMBER kMatrix<T, H, W> subtract(const kMatrix<T, H, W> &lhs, const kMatrix<T, H, W> &rhs,
                                               kMatrix<T, H, W> &out)
{
    T* out_buffer = out.buffer;
    for (int i = 0; i < lhs.nb_cols * lhs.nb_rows; i++)
    {
        out_buffer[i] = lhs.buffer[i] - rhs.buffer[i];
    }
    return out;
}

template <typename T, int H, int W>
CUDA_CALLABLE_MEMBER kMatrix<T, H, W> invert_matrix3(const kMatrix<T, H, W> &mat)
{
    auto inv_buf = invert_matrix(mat.buffer, 3);
    return kMatrix(inv_buf, 3, 3);
}

template <typename T, int H, int W, int S>
CUDA_CALLABLE_MEMBER kMatrix<T, H, S> matmul(const kMatrix<T, H, W> &lhs, const kMatrix<T, W, S> &rhs, 
                                kMatrix<T, H, S> &out)
{
    auto &out_buf = out.buffer;
    auto &m1 = lhs.buffer;
    auto &m2 = rhs.buffer;
    auto mult = matmul(m1, m2, out_buf, lhs.nb_rows, lhs.nb_cols, rhs.nb_cols);

    return out;
}


CUDA_CALLABLE_MEMBER float* get_adjugate_matrix(float* mat, float* out, size_t len);

}