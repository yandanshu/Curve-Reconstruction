#ifndef WARP_POINTS_HPP
#define WARP_POINTS_HPP

#include <vector>
#include <opencv2/opencv.hpp>

void warp_points(const std::vector<cv::Point2f>& points,
                 const cv::Mat& _map,
                 std::vector<cv::Point2f>& mapped_points);

#endif
