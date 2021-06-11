#pragma once
#include <Eigen/Dense>
#include <iostream>

float compute_det(Eigen::Matrix4d mat);

//Eigen::Matrix4d invert_matrix4(const Eigen::Matrix4d& mat);
Eigen::Matrix3d invert_matrix3(const Eigen::Matrix3d &mat);
std::vector<double> invert_matrix(const std::vector<double> &mat, size_t len);
double compute_det(const std::vector<double> &mat, size_t len);
Eigen::MatrixXd vec_to_eigen(std::vector<double> vec, size_t n_row, size_t n_col);

std::vector<double> matmul(const std::vector<double> &lhs, const std::vector<double> &rhs, size_t n, size_t p, size_t m);

Eigen::MatrixXd matmul(const Eigen::MatrixXd &lhs, const Eigen::MatrixXd &rhs);



std::vector<double> get_adjugate_matrix(const std::vector<double> &mat, size_t len);