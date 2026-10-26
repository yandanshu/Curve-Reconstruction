#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>

#include "config.hpp"

class Camera
{
  public:

    Camera();

    void load_intrinsics(const std::string& file_path, int camera_id, const double scale);

    void liftProjective(const Eigen::Vector2d& point, Eigen::Vector3d& point3D) const;

    void project(const Eigen::Vector3d& point3D, Eigen::Vector2d& point) const;

    cv::Mat undistMap1, undistMap2;

    double fx, fy, cx, cy;
    double k1, k2, p1, p2;
    Eigen::Matrix3d K;

    Eigen::Matrix3d RIC;
    Eigen::Vector3d TIC;

    int img_width;
    int img_height;

  private:

    void initUndistortRectifyMap();
};

#endif
