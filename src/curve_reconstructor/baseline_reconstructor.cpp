#include "baseline_reconstructor.hpp"
#include "pointcloud_to_nurbs.hpp"
#include "recon_config.hpp"
#include "tic_toc.h"
#include <iomanip>
#include <iomanip>

ControlPlane::ControlPlane(const Eigen::Vector3d& PO, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2)
    : PO_(PO), P1_(P1), P2_(P2) {}

bool ControlPlane::is_within_plane(const Eigen::Vector3d& p_int) const
{

    Eigen::Vector3d v1 = P1_ - PO_;
    Eigen::Vector3d v2 = P2_ - PO_;
    Eigen::Vector3d vp = p_int - PO_;

    double cos_alpha = v1.dot(vp) / (v1.norm() * vp.norm());
    double cos_beta  = v2.dot(vp) / (v2.norm() * vp.norm());
    double cos_gamma = v1.dot(v2) / (v1.norm() * v2.norm());

    double alpha = std::acos(cos_alpha);
    double beta  = std::acos(cos_beta);
    double gamma = std::acos(cos_gamma);

    return (alpha <= gamma && beta <= gamma);
}

bool ControlPlane::intersect_with_ray(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, Eigen::Vector3d& p_int) const
{

    Eigen::Vector3d n = (P1_ - PO_).cross(P2_ - PO_);

    Eigen::Vector3d dir = p1 - p0;

    double denom = n.dot(dir);
    if (std::abs(denom) < 1e-8)
        return false;

    double t = n.dot(PO_ - p0) / denom;

    if (t < 0)
        return false;

    p_int = p0 + t * dir;

    if(!is_within_plane(p_int)) return false;

    return true;
}

void BaselineReconstructor::save_point_clouds(const std::vector<Eigen::Vector3d>& points, const std::string& output_path) const
{
    if(points.empty()) return;

    cv::Scalar color = cv::Scalar(255, 0, 0);
    uint8_t b = color[0];
    uint8_t g = color[1];
    uint8_t r = color[2];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(const auto& point : points)
    {
        pcl::PointXYZRGB pcl_point;
        pcl_point.x = point.x();
        pcl_point.y = point.y();
        pcl_point.z = point.z();
        pcl_point.r = r;
        pcl_point.g = g;
        pcl_point.b = b;
        cloud->points.push_back(pcl_point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::PLYWriter writer;
    writer.write(output_path, *cloud, true);
    std::cout << cloud->points.size() << " points saved to " << output_path << std::endl;
}

BaselineReconstructor::BaselineReconstructor(const Curve& curve1, const Curve& curve2,
                         const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2,
                         const Eigen::Vector3d& t1, const Eigen::Vector3d& t2,
                         const Camera& camera1, const Camera& camera2)
    : curve1_(curve1), curve2_(curve2), R1_(R1), R2_(R2), t1_(t1), t2_(t2), camera1_(camera1), camera2_(camera2) {}

void BaselineReconstructor::set_s_grid_count(int count)
{
    s_grid_count_ = count;
}

void BaselineReconstructor::set_delta_s(double delta_s)
{
    delta_s_ = delta_s;
}

Eigen::Vector3d BaselineReconstructor::get_point_on_ruled_surface(const double& param1, const double& param2,
                                                        const Curve& curve, const Camera& camera,
                                                        const Eigen::Matrix3d& R, const Eigen::Vector3d t)
{
    LNLib::XYZ _P = LNLib::NurbsCurve::GetPointOnCurve(curve.nurbs, param1);
    Eigen::Vector3d _P_3d;
    camera.liftProjective(Eigen::Vector2d(_P.X(), _P.Y()), _P_3d);
    Eigen::Vector3d P = R * _P_3d + t;
    Eigen::Vector3d PO = t;

    return (1 - param2) * PO + param2 * P;
}

void BaselineReconstructor::calculate_initial_values()
{
    initial_values_.clear();

    std::vector<LNLib::XYZW> cps2 = curve2_.nurbs.ControlPoints;

    if(cps2.size() < 2) return;

    std::vector<double> kv2 = curve2_.nurbs.KnotVector;

    intersections_.clear();

    Eigen::Vector3d PO_2 = t2_;
    std::vector<ControlPlane> ctrl_planes;
    for(int i = 0; i < cps2.size() - 1; ++i)
    {
        Eigen::Vector3d _p_3d;
        camera2_.liftProjective(Eigen::Vector2d(cps2[i].WX(), cps2[i].WY()), _p_3d);
        Eigen::Vector3d P1_2 = R2_ * _p_3d + t2_;
        camera2_.liftProjective(Eigen::Vector2d(cps2[i + 1].WX(), cps2[i + 1].WY()), _p_3d);
        Eigen::Vector3d P2_2 = R2_ * _p_3d + t2_;

        ControlPlane cp(PO_2, P1_2, P2_2);
        ctrl_planes.emplace_back(cp);
    }

    Eigen::Vector3d PO_1 = t1_;

    const auto run_s = [&](double s) {
        LNLib::XYZ _P = LNLib::NurbsCurve::GetPointOnCurve(curve1_.nurbs, s);
        Eigen::Vector3d _p_3d;
        camera1_.liftProjective(Eigen::Vector2d(_P.X(), _P.Y()), _p_3d);
        Eigen::Vector3d P_1 = R1_ * _p_3d + t1_;

        bool found_flag = false;

        for(int i = 0; i < ctrl_planes.size(); ++i)
        {

            Eigen::Vector3d P_int;
            if(!ctrl_planes[i].intersect_with_ray(PO_1, P_1, P_int)) continue;

            intersections_.emplace_back(P_int);
            found_flag = true;

            double s_i = s;
            double u_i = (kv2[i] + kv2[i + 1]) * 0.5;
            double v_i = (P_int - ctrl_planes[i].PO_).norm() / (ctrl_planes[i].P1_ - ctrl_planes[i].PO_).norm();
            double t_i = v_i;

            initial_values_.emplace_back(std::array<double, 4>{s_i, u_i, v_i, t_i});

            break;
        }
    };

    if (s_grid_count_ > 1) {
        for (int i = 0; i < s_grid_count_; ++i) {
            const double s = (s_grid_count_ > 1)
                ? static_cast<double>(i) / static_cast<double>(s_grid_count_ - 1)
                : 0.0;
            run_s(s);
        }
    } else {
        for (double s = 0.0; s <= 1.0 + 1e-9; s += delta_s_)
            run_s(s);
    }

    if(initial_values_.empty()) return;

    init_flag_ = true;

    const double ds_log = (s_grid_count_ > 1)
        ? 1.0 / (s_grid_count_ - 1)
        : delta_s_;

    initial_points_.clear();
    initial_points_.reserve(initial_values_.size());
    for(const auto& values : initial_values_)
    {
        double s = values[0], u = values[1], v = values[2], t = values[3];
        Eigen::Vector3d P1 = get_point_on_ruled_surface(s, t, curve1_, camera1_, R1_, t1_);
        Eigen::Vector3d P2 = get_point_on_ruled_surface(u, v, curve2_, camera2_, R2_, t2_);
        Eigen::Vector3d P = (P1 + P2) * 0.5;
        initial_points_.emplace_back(P);
    }

    return;
}

Eigen::Vector3d BaselineReconstructor::get_residual(const double& s, const double& u, const double& v, const double& t)
{

    Eigen::Vector3d P1 = get_point_on_ruled_surface(s, t, curve1_, camera1_, R1_, t1_);

    Eigen::Vector3d P2 = get_point_on_ruled_surface(u, v, curve2_, camera2_, R2_, t2_);

    return P2 - P1;
}

Eigen::Matrix3d BaselineReconstructor::get_Jacobian(const double& s, const double& u, const double& v, const double& t, const double& delta)
{
    Eigen::Matrix3d J;
    Eigen::Vector3d R0 = get_residual(s, u, v, t);

    Eigen::Vector3d Ru = get_residual(s, u + delta, v, t);
    J.col(0) = (Ru - R0) / delta;

    Eigen::Vector3d Rv = get_residual(s, u, v + delta, t);
    J.col(1) = (Rv - R0) / delta;

    Eigen::Vector3d Rt = get_residual(s, u, v, t + delta);
    J.col(2) = (Rt - R0) / delta;

    return J;
}

std::vector<Eigen::Vector3d> BaselineReconstructor::fit_nurbs_legacy_sample(
    const std::vector<Eigen::Vector3d>& points, const LNLib::LN_NurbsCurve& nurbs)
{
    std::vector<Eigen::Vector3d> outputs;
    if (points.size() < 2) return outputs;

    outputs.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const double s = static_cast<double>(i) / static_cast<double>(points.size() - 1);
        const LNLib::XYZ p = LNLib::NurbsCurve::GetPointOnCurve(nurbs, s);
        if (std::isnan(p.X()) || std::isnan(p.Y()) || std::isnan(p.Z()))
            return {};
        outputs.emplace_back(Eigen::Vector3d(p.X(), p.Y(), p.Z()));
    }
    return outputs;
}

bool BaselineReconstructor::fit_nurbs_curve(LNLib::LN_NurbsCurve& nurbs,
                                            int& ctrl_used,
                                            int ctrl_override) const
{
    ctrl_used = 0;
    if (optimized_points_.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1)) return false;

    return fit_points_to_nurbs(optimized_points_, nurbs, ctrl_used,
                               camera1_.img_height, ctrl_override);
}

void BaselineReconstructor::trace_surface_intersection()
{
    if(!init_flag_) return;

    optimized_values_.clear();

    double tol = 1e-6;
    int max_iter = 50;

    for(const auto& values : initial_values_)
    {
        double s = values[0], u = values[1], v = values[2], t = values[3];
        bool success_flag = false;
        for(int i = 0; i < max_iter; ++i)
        {
            Eigen::Vector3d res = get_residual(s, u, v, t);
            double norm_res = res.norm();

            if(norm_res <= tol)
            {

                success_flag = true;
                break;
            }

            Eigen::Matrix3d J = get_Jacobian(s, u, v, t);

            Eigen::Vector3d delta = J.fullPivLu().solve(-res);

            u += delta(0);
            v += delta(1);
            t += delta(2);

            u = std::clamp(u, 0.0, 1.0);
        }

        if(!success_flag)
        {

            continue;
        }

        optimized_values_.emplace_back(std::array<double, 4>{s, u, v, t});
    }

    if(optimized_values_.empty()) return;
    opt_flag_ = true;

    optimized_points_.clear();

    for(const auto& values : optimized_values_)
    {
        double s = values[0], u = values[1], v = values[2], t = values[3];
        Eigen::Vector3d P1 = get_point_on_ruled_surface(s, t, curve1_, camera1_, R1_, t1_);
        Eigen::Vector3d P2 = get_point_on_ruled_surface(u, v, curve2_, camera2_, R2_, t2_);
        Eigen::Vector3d P = (P1 + P2) * 0.5;

        optimized_points_.emplace_back(P);
    }

    if(optimized_points_.empty()) return;

    return;
}

bool BaselineReconstructor::process_up_to_optimization()
{
    assert(curve1_.is_curved == curve2_.is_curved);

    init_flag_ = false;
    opt_flag_ = false;
    fit_flag_ = false;

    calculate_initial_values();
    trace_surface_intersection();

    if (!init_flag_ || !opt_flag_) return false;
    if (optimized_points_.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1)) return false;
    return true;
}

bool BaselineReconstructor::process()
{
    if (!process_up_to_optimization()) return false;

    LNLib::LN_NurbsCurve nurbs;
    int ctrl_used = 0;
    if (!fit_nurbs_curve(nurbs, ctrl_used, -1)) return false;

    nurbs_points_ = fit_nurbs_legacy_sample(optimized_points_, nurbs);
    if (nurbs_points_.empty()) return false;

    fit_flag_ = true;
    return true;
}

bool BaselineReconstructor::get_results(std::vector<Eigen::Vector3d>& intersections,
                              std::vector<Eigen::Vector3d>& initial_points,
                              std::vector<Eigen::Vector3d>& optimized_points,
                              std::vector<Eigen::Vector3d>& nurbs_points)
{
    if(intersections_.empty() || initial_points_.empty() || optimized_points_.empty() || nurbs_points_.empty())
        return 0;

    intersections = intersections_;
    initial_points = initial_points_;
    optimized_points = optimized_points_;
    nurbs_points = nurbs_points_;

    return 1;
}
