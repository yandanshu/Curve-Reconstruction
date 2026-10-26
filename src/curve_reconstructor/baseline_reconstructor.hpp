#ifndef BASELINE_RECONSTRUCTOR_HPP
#define BASELINE_RECONSTRUCTOR_HPP

#include <iostream>
#include "curve.hpp"
#include "NurbsCurve.h"
#include <Eigen/Dense>
#include "camera.hpp"
#include <pcl/io/ply_io.h>

class ControlPlane
{
  public:
    Eigen::Vector3d PO_, P1_, P2_;
    ControlPlane(const Eigen::Vector3d& PO, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2);
    bool intersect_with_ray(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, Eigen::Vector3d& p_int) const;

  private:
    bool is_within_plane(const Eigen::Vector3d& p_int) const;
};

class BaselineReconstructor
{
  public:
    BaselineReconstructor(const Curve& curve1, const Curve& curve2,
                const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2,
                const Eigen::Vector3d& t1, const Eigen::Vector3d& t2,
                const Camera& camera1, const Camera& camera2);

    void set_s_grid_count(int count);
    void set_delta_s(double delta_s);

    bool process();
    bool process_up_to_optimization();
    bool fit_nurbs_curve(LNLib::LN_NurbsCurve& nurbs, int& ctrl_used, int ctrl_override = -1) const;

    bool get_results(std::vector<Eigen::Vector3d>& intersections,
                     std::vector<Eigen::Vector3d>& initial_points,
                     std::vector<Eigen::Vector3d>& optimized_points,
                     std::vector<Eigen::Vector3d>& nurbs_points);

    const std::vector<Eigen::Vector3d>& optimized_points_ref() const { return optimized_points_; }

  private:
    void calculate_initial_values();
    void trace_surface_intersection();
    void save_point_clouds(const std::vector<Eigen::Vector3d>& points, const std::string& output_path) const;
    Eigen::Vector3d get_point_on_ruled_surface(const double& param1, const double& param2,
                                               const Curve& curve, const Camera& camera,
                                               const Eigen::Matrix3d& R, const Eigen::Vector3d t);
    Eigen::Vector3d get_residual(const double& s, const double& u, const double& v, const double& t);
    Eigen::Matrix3d get_Jacobian(const double& s, const double& u, const double& v, const double& t, const double& delta = 1e-6);
    std::vector<Eigen::Vector3d> fit_nurbs_legacy_sample(const std::vector<Eigen::Vector3d>& points,
                                                         const LNLib::LN_NurbsCurve& nurbs);

    Curve curve1_, curve2_;
    Eigen::Matrix3d R1_, R2_;
    Eigen::Vector3d t1_, t2_;
    Camera camera1_, camera2_;

    int s_grid_count_ = -1;
    double delta_s_ = 0.01;

    bool init_flag_ = false, opt_flag_ = false, fit_flag_ = false;
    std::vector<std::array<double, 4>> initial_values_, optimized_values_;

    std::vector<Eigen::Vector3d> intersections_, initial_points_, optimized_points_, nurbs_points_;
};

#endif
