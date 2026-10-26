#include "pointcloud_to_nurbs.hpp"
#include "recon_config.hpp"
#include <iostream>

std::vector<Eigen::Vector3d> SortPointCloudByPrincipalDirection(const std::vector<Eigen::Vector3d>& points)
{
    if (points.empty()) return {};

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : points)
        mean += p;
    mean /= static_cast<double>(points.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        Eigen::Vector3d centered = p - mean;
        cov += centered * centered.transpose();
    }
    cov /= static_cast<double>(points.size() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d eigen_values = solver.eigenvalues();
    Eigen::Matrix3d eigen_vectors = solver.eigenvectors();

    int max_idx;
    eigen_values.maxCoeff(&max_idx);
    Eigen::Vector3d principal_dir = eigen_vectors.col(max_idx).normalized();

    std::vector<std::pair<double, Eigen::Vector3d>> projected_points;
    projected_points.reserve(points.size());
    for (const auto& p : points) {
        double projection = (p - mean).dot(principal_dir);
        projected_points.emplace_back(projection, p);
    }

    std::sort(projected_points.begin(), projected_points.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<Eigen::Vector3d> sorted_points;
    sorted_points.reserve(points.size());
    for (const auto& pair : projected_points)
        sorted_points.push_back(pair.second);

    return sorted_points;
}

bool fit_points_to_nurbs(const std::vector<Eigen::Vector3d>& points,
                         LNLib::LN_NurbsCurve& nurbs,
                         int& ctrl_used,
                         int img_height,
                         int ctrl_override)
{
    ctrl_used = 0;
    if (points.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1)) return false;

    std::vector<Eigen::Vector3d> sorted = SortPointCloudByPrincipalDirection(points);
    std::vector<LNLib::XYZ> xyz;
    xyz.reserve(sorted.size());
    for (const auto& p : sorted)
        xyz.emplace_back(LNLib::XYZ(p.x(), p.y(), p.z()));

    const int n = static_cast<int>(xyz.size());
    int ctrl_point_num = 0;

    if (ctrl_override >= 0) {
        ctrl_point_num = std::clamp(ctrl_override, recon_cfg().p_nurbs + 1, n);
        if (!LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, xyz, ctrl_point_num, nurbs))
            return false;
        ctrl_used = ctrl_point_num;
        return true;
    }

    LNLib::LN_NurbsCurve coarse;
    if (!LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, xyz, 10, coarse))
        return false;

    double rough_length = LNLib::NurbsCurve::ApproximateLength(coarse, LNLib::IntegratorType::GaussLegendre);
    ctrl_point_num = static_cast<int>(std::round(rough_length / static_cast<double>(img_height) * 100.0));
    ctrl_point_num = std::clamp(ctrl_point_num, recon_cfg().p_nurbs + 1, n);

    if (!LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, xyz, ctrl_point_num, nurbs))
        return false;
    ctrl_used = ctrl_point_num;
    return true;
}
