//
// Created by alexandre on 11/06/2021.
//

#include "../include/matrix_tools.hpp"


std::vector<double> matmul(const std::vector<double> &lhs, const std::vector<double> &rhs, size_t n, size_t p, size_t m)
{
    auto out = std::vector<double>(n * m);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            double sum = 0;
            for (size_t k = 0; k < p; k++)
            {
                sum += lhs[k + i * p] * rhs[k * m + j];
            }
            out[j + i * m] = sum;
        }
    }
    return out;
}


std::vector<double> eigen_to_vec(const Eigen::MatrixXd &mat)
{
    auto out = std::vector<double>(mat.cols() * mat.rows());

    for (int i = 0; i < mat.cols(); i++)
        for (int j = 0; j < mat.rows(); j++)
            out[i + j * mat.cols()] = mat(j, i);

    return out;
}

Eigen::MatrixXd matmul(const Eigen::MatrixXd &lhs, const Eigen::MatrixXd &rhs)
{
    if (lhs.cols() != rhs.rows())
    {
        throw std::runtime_error("psa la meme dimension");
    }
    auto m1 = eigen_to_vec(lhs);
    auto m2 = eigen_to_vec(rhs);
    auto mult = matmul(m1, m2, lhs.rows(), lhs.cols(), rhs.cols());

    auto out = Eigen::MatrixXd(lhs.rows(), rhs.cols());

    return vec_to_eigen(mult, lhs.rows(), rhs.cols());
}

Eigen::MatrixXd vec_to_eigen(std::vector<double> vec, size_t n_row, size_t n_col)
{
    auto out = Eigen::MatrixXd(n_row, n_col);

    for (int i = 0; i < out.rows(); i++)
    {
        for (int j = 0; j < out.cols(); j++)
        {
            out(i, j) = vec[i * out.cols() + j];
        }
    }
    return out;
}


std::vector<double> invert_matrix(const std::vector<double> &mat, size_t len)
{
    auto det = compute_det(mat, len);
    auto adj = get_adjugate_matrix(mat, len);
    for (auto &elm : adj)
        elm = elm / det;

    return adj;
}


double compute_det(const std::vector<double> &mat, size_t len)
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


std::vector<double> get_adjugate_matrix(const std::vector<double> &mat, size_t len)
{
    if (len == 4)
    {
        std::vector<double> out;
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

        out.push_back(a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43
                      - a24 * a33 * a42 - a23 * a32 * a44 - a22 * a34 * a43);
        out.push_back(-a12 * a33 * a44 - a13 * a34 * a42 - a14 * a32 * a43
                      + a14 * a33 * a42 + a13 * a32 * a44 + a12 * a34 * a43);
        out.push_back(a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43
                      - a14 * a23 * a42 - a13 * a22 * a44 - a12 * a24 * a43);
        out.push_back(-a12 * a23 * a34 - a13 * a24 * a32 - a14 * a22 * a33
                      + a14 * a23 * a32 + a13 * a22 * a34 + a12 * a24 * a33);
        out.push_back(-a21 * a31 * a44 - a23 * a34 * a41 - a24 * a31 * a43
                      + a24 * a33 * a41 + a23 * a31 * a44 + a21 * a34 * a43);
        out.push_back(a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43
                      - a14 * a33 * a41 - a13 * a31 * a44 - a11 * a34 * a43);


        out.push_back(-a11 * a23 * a44 - a13 * a24 * a41 - a14 * a21 * a43
                      + a14 * a23 * a41 + a13 * a21 * a44 + a11 * a24 * a43);


        out.push_back(a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33
                      - a14 * a23 * a31 - a13 * a21 * a34 - a11 * a24 * a33);
        out.push_back(a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42
                      - a24 * a32 * a41 - a22 * a31 * a44 - a21 * a34 * a42);
        out.push_back(-a11 * a32 * a44 - a12 * a34 * a41 - a14 * a31 * a42
                      + a14 * a32 * a41 + a12 * a31 * a44 + a11 * a34 * a42);
        out.push_back(a11 *
                      a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42
                      - a14 * a22 * a41 - a12 * a21 * a44 - a11 * a24 * a42);
        out.push_back(-a11 * a22 * a34 - a12 * a24 * a31 - a14 * a21 * a32
                      + a14 * a22 * a31 + a12 * a21 * a34 + a11 * a24 * a32);
        out.push_back(-a21 * a32 * a43 - a22 * a33 * a41 - a23 * a31 * a42
                      + a23 * a32 * a41 + a22 * a31 * a43 + a21 * a33 * a42);


        out.push_back(a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42
                      - a13 * a32 * a41 - a12 * a31 * a43 - a11 * a33 * a42);

        out.push_back(-a11 * a22 * a43 - a12 * a23 * a41 - a13 * a21 * a42
                      + a13 * a22 * a41 + a12 * a21 * a43 + a11 * a23 * a42);


        out.push_back(a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
                      - a13 * a22 * a31 - a12 * a21 * a33 - a11
                                                            * a23 * a32);
        return out;
    } else if (len == 3)
    {
        std::vector<double> out;
        auto &a = mat[0];
        auto &b = mat[1];
        auto &c = mat[2];
        auto &d = mat[3];
        auto &e = mat[4];
        auto &f = mat[5];
        auto &g = mat[6];
        auto &h = mat[7];
        auto &i = mat[8];

        out.push_back(e * i - f * h);
        out.push_back(-b * i + h * c);
        out.push_back(b * f - e * c);
        out.push_back(-d * i + g * f);
        out.push_back(a * i - c * g);
        out.push_back(-a * f + d * c);
        out.push_back(d * h - g * e);
        out.push_back(-a * h + g * b);
        out.push_back(a * e - d * b);
        return out;
    }
}


//Eigen::Matrix4d invert_matrix4(const Eigen::Matrix4d &mat)
//{
//    auto vec = eigen_to_vec(mat);
//    auto inv_vec = invert_matrix(vec, 4);
//    return Eigen::Matrix4d(inv_vec.data());
//}

Eigen::Matrix3d invert_matrix3(const Eigen::Matrix3d &mat)
{
    auto vec = eigen_to_vec(mat);
    auto inv_vec = invert_matrix(vec, 3);
    return Eigen::Matrix3d(inv_vec.data());
}
