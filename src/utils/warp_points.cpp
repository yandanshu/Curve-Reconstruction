#include "warp_points.hpp"

void warp_points(const std::vector<cv::Point2f>& points,
                 const cv::Mat& _map,
                 std::vector<cv::Point2f>& mapped_points)
{
    mapped_points.reserve(points.size());

    for (const auto& pt : points)
    {

        int x0 = static_cast<int>(pt.x);
        int y0 = static_cast<int>(pt.y);
        int x1 = std::min(x0 + 1, _map.cols - 1);
        int y1 = std::min(y0 + 1, _map.rows - 1);

        float dx = pt.x - x0;
        float dy = pt.y - y0;

        cv::Vec2f p00 = _map.at<cv::Vec2f>(y0, x0);
        cv::Vec2f p10 = _map.at<cv::Vec2f>(y0, x1);
        cv::Vec2f p01 = _map.at<cv::Vec2f>(y1, x0);
        cv::Vec2f p11 = _map.at<cv::Vec2f>(y1, x1);

        cv::Point2f warped_pt =
            (1 - dx) * (1 - dy) * cv::Point2f(p00[0], p00[1]) +
            dx       * (1 - dy) * cv::Point2f(p10[0], p10[1]) +
            (1 - dx) * dy       * cv::Point2f(p01[0], p01[1]) +
            dx       * dy       * cv::Point2f(p11[0], p11[1]);

        mapped_points.push_back(warped_pt);
    }
}
