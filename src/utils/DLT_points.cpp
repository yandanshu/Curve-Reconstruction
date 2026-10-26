#include "DLT_points.hpp"

void DLT_points(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2,
                std::vector<Eigen::Vector3d>& pts_3d)
{

    pts_3d.clear();

    Eigen::Matrix3d Rcw1 = R1.transpose();
    Eigen::Vector3d tcw1 = -R1.transpose() * t1;

    Eigen::Matrix3d Rcw2 = R2.transpose();
    Eigen::Vector3d tcw2 = -R2.transpose() * t2;

    Eigen::Matrix<double,3,4> P1, P2;
    P1.block<3,3>(0,0) = Rcw1; P1.col(3) = tcw1; P1 = K1 * P1;
    P2.block<3,3>(0,0) = Rcw2; P2.col(3) = tcw2; P2 = K2 * P2;

    for(size_t i=0; i<pts1.size(); ++i)
    {
        Eigen::Matrix4d A;
        const cv::Point2f& pt1 = pts1[i];
        const cv::Point2f& pt2 = pts2[i];

        A.row(0) = pt1.x * P1.row(2) - P1.row(0);
        A.row(1) = pt1.y * P1.row(2) - P1.row(1);
        A.row(2) = pt2.x * P2.row(2) - P2.row(0);
        A.row(3) = pt2.y * P2.row(2) - P2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X_hom = svd.matrixV().col(3);

        pts_3d.emplace_back(X_hom.head<3>() / X_hom(3));

    }
}
