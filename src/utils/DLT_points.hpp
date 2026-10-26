#ifndef DLT_POINTS_HPP
#define DLT_POINTS_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void DLT_points(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                std::vector<Eigen::Vector3d>& pts_3d);

#endif
