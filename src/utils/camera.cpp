#include "camera.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

Camera::Camera()
{
    RIC << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;

    TIC << 0, 0, 0;

    k1 = 0; k2 = 0; p1 = 0; p2 = 0;
}

void Camera::load_intrinsics(const std::string& file_path, const int cam_id, const double scale)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open camera file: " + file_path);
    }

    std::string line;
    bool found = false;
    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);

        int _camera_id, _img_width, _img_height;
        std::string _camera_type;
        double _fx, _fy, _cx, _cy;
        if (!(ss >> _camera_id >> _camera_type >> _img_width >> _img_height
                 >> _fx >> _fy >> _cx >> _cy)) {
            throw std::runtime_error("Invalid camera record in " + file_path);
        }

        if(_camera_id == cam_id && _camera_type == "PINHOLE")
        {
            fx = _fx; fy = _fy; cx = _cx; cy = _cy;
            img_width = _img_width; img_height = _img_height;

            fx *= scale; fy *= scale; cx *= scale; cy *= scale;
            K << fx, 0, cx,
                0, fy, cy,
                0,  0, 1;

            img_width = std::round(img_width * scale); img_height = std::round(img_height * scale);

            found = true;
            break;
        }
    }
    if(!found)
    {
        throw std::runtime_error("PINHOLE camera " + std::to_string(cam_id) + " not found in " + file_path);
    }

    initUndistortRectifyMap();
}

void Camera::initUndistortRectifyMap()
{

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                                    0, fy, cy,
                                                    0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat newCameraMatrix = cameraMatrix.clone();

    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs,
                                R,
                                newCameraMatrix,
                                cv::Size(img_width, img_height),
                                CV_16SC2,
                                undistMap1,
                                undistMap2);
}

void Camera::liftProjective(const Eigen::Vector2d& point, Eigen::Vector3d& point3D) const
{

    double x = (point.x() - cx) / fx;
    double y = (point.y() - cy) / fy;

    point3D << x, y, 1.0;
}

void Camera::project(const Eigen::Vector3d& point3D, Eigen::Vector2d& point) const
{

    double x = point3D.x() / point3D.z();
    double y = point3D.y() / point3D.z();

    point.x() = fx * x + cx;
    point.y() = fy * y + cy;
}
