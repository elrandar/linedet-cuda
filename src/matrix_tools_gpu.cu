//
// Created by alexandre on 11/06/2021.
//

#include "../include/matrix_tools_gpu.cuh"

namespace kalman_gpu
{
__device__ float* matmul(float* lhs, float* rhs, float* out, size_t n, size_t p,
                           size_t m)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            float sum = 0;
            for (size_t k = 0; k < p; k++)
            {
                sum += lhs[k + i * p] * rhs[k * m + j];
            }
            out[j + i * m] = sum;
        }
    }
    return out;
}

__device__ float* matmul(const float* lhs, const float* rhs, float* out, const size_t n, const size_t p,
                           const size_t m)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            float sum = 0;
            for (size_t k = 0; k < p; k++)
            {
                sum += lhs[k + i * p] * rhs[k * m + j];
            }
            out[j + i * m] = sum;
        }
    }
    return out;
}


__device__ float* invert_matrix(float* mat, float* out, size_t len)
{
    auto det = compute_det(mat, len);
    get_adjugate_matrix(mat, out, len);

    for (int i = 0; i < len * len; i++)
    {
        out[i] = out[i] / det;
    }

    return out;
}


__device__ float compute_det(float* mat, size_t len)
{
    if (len == 4)
    {
        auto &a = mat[0];
        auto &b = mat[1];
        auto &c = mat[2];
        auto &d = mat[3];
        auto &e = mat[4];
        auto &f = mat[5];
        auto &g = mat[6];
        auto &h = mat[7];
        auto &i = mat[8];
        auto &j = mat[9];
        auto &k = mat[10];
        auto &l = mat[11];
        auto &m = mat[12];
        auto &n = mat[13];
        auto &o = mat[14];
        auto &p = mat[15];

        return a * f * k * p - a * f * l * o - a * g * j * p + a * g * l * n + a * h * j * o - a * h * k * n -
               b * e * k * p + b * e * l * o + b * g * i * p - b * g * l * m - b * h * i * o + b * h * k * m +
               c * e * j * p - c * e * l * n - c * f * i * p + c * f * l * m + c * h * i * n - c * h * j * m -
               d * e * j * o + d * e * k * n + d * f * i * o - d * f * k * m - d * g * i * n + d * g * j * m;

    } else if (len == 3)
    {
        auto &a = mat[0];
        auto &b = mat[1];
        auto &c = mat[2];
        auto &d = mat[3];
        auto &e = mat[4];
        auto &f = mat[5];
        auto &g = mat[6];
        auto &h = mat[7];
        auto &i = mat[8];

        return a * e * i - a * f * h + b * f * g - b * d * i + c * d * h - c * e * g;

    } else
        return mat[0] * mat[3] - mat[1] * mat[2];
}


__device__ float* get_adjugate_matrix(float* mat, float* out, size_t len)
{
    if (len == 4)
    {
        auto &a11 = mat[0];
        auto &a12 = mat[1];
        auto &a13 = mat[2];
        auto &a14 = mat[3];
        auto &a21 = mat[4];
        auto &a22 = mat[5];
        auto &a23 = mat[6];
        auto &a24 = mat[7];
        auto &a31 = mat[8];
        auto &a32 = mat[9];
        auto &a33 = mat[10];
        auto &a34 = mat[11];
        auto &a41 = mat[12];
        auto &a42 = mat[13];
        auto &a43 = mat[14];
        auto &a44 = mat[15];

        out[0] = (a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43
                      - a24 * a33 * a42 - a23 * a32 * a44 - a22 * a34 * a43);
        out[1] = (-a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43
                      + a14 * a33 * a42 + a13 * a32 * a44 + a12 * a34 * a43);
        out[2] = (a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43
                      - a14 * a23 * a42 - a13 * a22 * a44 - a12 * a24 * a43);
        out[3] = (-a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33
                      + a14 * a23 * a32 + a13 * a22 * a34 + a12 * a24 * a33);
        out[4] = (-a21 * a31 * a44 - a23 * a34 * a41 - a24 * a31 * a43
                      + a24 * a33 * a41 + a23 * a31 * a44 + a21 * a34 * a43);
        out[5] = (a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43
                      - a14 * a33 * a41 - a13 * a31 * a44 - a11 * a34 * a43);


        out[6] = (-a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43
                      + a14 * a23 * a41 + a13 * a21 * a44 + a11 * a24 * a43);


        out[7] = (a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33
                      - a14 * a23 * a31 - a13 * a21 * a34 - a11 * a24 * a33);
        out[8] = (a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42
                      - a24 * a32 * a41 - a22 * a31 * a44 - a21 * a34 * a42);
        out[9] = (-a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42
                      + a14 * a32 * a41 + a12 * a31 * a44 + a11 * a34 * a42);
        out[10] = (a11 *
                      a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42
                      - a14 * a22 * a41 - a12 * a21 * a44 - a11 * a24 * a42);
        out[11] = (-a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32
                      + a14 * a22 * a31 + a12 * a21 * a34 + a11 * a24 * a32);
        out[12] = (-a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42
                      + a23 * a32 * a41 + a22 * a31 * a43 + a21 * a33 * a42);


        out[13] = (a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42
                      - a13 * a32 * a41 - a12 * a31 * a43 - a11 * a33 * a42);

        out[14] = (-a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42
                      + a13 * a22 * a41 + a12 * a21 * a43 + a11 * a23 * a42);


        out[15] = (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
                      - a13 * a22 * a31 - a12 * a21 * a33 - a11
                                                            * a23 * a32);
        return out;
    } else if (len == 3)
    {
        auto &a = mat[0];
        auto &b = mat[1];
        auto &c = mat[2];
        auto &d = mat[3];
        auto &e = mat[4];
        auto &f = mat[5];
        auto &g = mat[6];
        auto &h = mat[7];
        auto &i = mat[8];

        out[0] = (e * i - f * h);
        out[1] = (-b * i + h * c);
        out[2] = (b * f - e * c);
        out[3] = (-d * i + g * f);
        out[4] = (a * i - c * g);
        out[5] = (-a * f + d * c);
        out[6] = (d * h - g * e);
        out[7] = (-a * h + g * b);
        out[8] = (a * e - d * b);
        return out;
    }
    return nullptr;
}
}
