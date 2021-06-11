////    compute_det(Eigen::Matrix4d::Identity());
//std::cout << compute_det(std::vector<double>({1, 7, 9, 3,
//                                              0, 7, 9, 0,
//                                              8, 9, 1, 5,
//                                              5, 3, 0, 9}), 4);
//
//for (auto & elm : invert_matrix(std::vector<double>({1, 8, 6, 0,
//                                                     6, 2, 0, 8,
//                                                     4, 3, 6, 0,
//                                                     9, 2, 5, 3}), 4))
//{
//std::cout << elm << " ";
//}
//
//std::cout << '\n';
//auto mat = Eigen::Matrix<double, 4, 4>();
//mat << 1, 8, 6, 0,
//6, 2, 0, 8,
//4, 3, 6, 0,
//9, 2, 5, 3;
//std::cout << invert_matrix(mat
//);

//    auto m1 = std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8, 9});
////    auto m2 = std::vector<double>({1, 2, 0 , 0, 1, 0, 0, 0, 1});
////
////    auto m3 = matmul(m1, m2, 3, 3, 3);
////    std::cout << Eigen::Matrix<double, 3, 3>(m3.data());
//    auto m1 = std::vector<double>({1, 2, 4, 5, 7, 8});
//    auto m2 = std::vector<double>({1, 2, 6 , 1, 0, 1, 0, 2});
//
//    auto m3 = matmul(m1, m2, 3, 2, 4);
//    std::cout << Eigen::Matrix<double, 3, 4>(m3.data());