#ifndef DEPTH_RBF_HPP
#define DEPTH_RBF_HPP

#include <vector>
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix3d skew(const Eigen::Vector3d &v);

double angle_difference(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);

void approximate_depth_with_GaussianRBF(const std::vector<Eigen::Vector3d>& p1, const std::vector<Eigen::Vector3d>& p2,
                        const std::vector<double>& s1, const std::vector<double>& s2,
                        const std::vector<double>& w1, const std::vector<double>& w2,
                        const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                        const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                        const int num_RBF,
                        Eigen::VectorXd& coeffs1, Eigen::VectorXd& coeffs2,
                        std::vector<double>& centers1, std::vector<double>& centers2,
                        std::vector<double>& sigmas1, std::vector<double>& sigmas2,
                        double& residual);

double get_approximated_depth(const std::vector<double>& centers, const std::vector<double>& sigmas,
                              const Eigen::VectorXd& coeffs, const double s);

#endif
