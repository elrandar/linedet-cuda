#pragma once
#include <Eigen/Dense>
#include <iostream>

float compute_det(Eigen::Matrix4d mat);

//Eigen::Matrix4d invert_matrix4(const Eigen::Matrix4d& mat);
Eigen::Matrix3d invert_matrix3(const Eigen::Matrix3d &mat);
std::vector<double> invert_matrix(const std::vector<double> &mat, size_t len);
double compute_det(const std::vector<double> &mat, size_t len);

std::vector<double> get_adjugate_matrix(const std::vector<double> &mat, size_t len);