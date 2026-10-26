#include "curve_reconstructor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>

#include "recon_config.hpp"
#include "warp_points.hpp"
#include "DLT_points.hpp"
#include "NurbsCurve.h"
#include "baseline_reconstructor.hpp"
#include "pointcloud_to_nurbs.hpp"
#include "depth_rbf.hpp"
#include "find_s.hpp"
#include "tic_toc.h"
#include <pcl/io/ply_io.h>

namespace {

constexpr double k_max_zhou_nurbs_arc_length = 10000.0;

void reset_resulted_feature_timing(ResultedFeature& rf, bool curve)
{
    rf.time_consumptions.clear();
    rf.overall_time_consumption = 0.0;
    rf.is_done = false;
    rf.is_curve = curve;
    rf.geometry = ResultGeometry::Empty;
    rf.fit_points.clear();
    rf.has_nurbs = false;
    rf.line.valid = false;
    rf.ctrl_point_num = 0;
}

void finalize_overall_time(ResultedFeature& rf)
{
    if (!rf.is_done) {
        rf.overall_time_consumption = 0.0;
        return;
    }
    rf.overall_time_consumption = 0.0;
    for (double t : rf.time_consumptions)
        rf.overall_time_consumption += t;
}

double average_or_placeholder(const std::vector<double>& values)
{
    if (values.empty())
        return RuntimeStats::kPlaceholder;
    double sum = 0.0;
    for (double v : values)
        sum += v;
    return sum / static_cast<double>(values.size());
}

std::vector<Eigen::Vector3d> sample_nurbs_by_ds(const LNLib::LN_NurbsCurve& nurbs, double ds)
{
    std::vector<Eigen::Vector3d> out;
    const double L = LNLib::NurbsCurve::ApproximateLength(nurbs, LNLib::IntegratorType::GaussLegendre);
    if (L <= 0.0 || ds <= 0.0) return out;

    const int table_n = std::max(64, static_cast<int>(std::ceil(L / ds)) * 4);

    std::vector<double> cum_len(static_cast<size_t>(table_n), 0.0);
    std::vector<double> params(static_cast<size_t>(table_n), 0.0);

    Eigen::Vector3d prev;
    {
        const LNLib::XYZ p0 = LNLib::NurbsCurve::GetPointOnCurve(nurbs, 0.0);
        prev = Eigen::Vector3d(p0.X(), p0.Y(), p0.Z());
    }
    for (int i = 1; i < table_n; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(table_n - 1);
        params[static_cast<size_t>(i)] = u;
        const LNLib::XYZ pi = LNLib::NurbsCurve::GetPointOnCurve(nurbs, u);
        const Eigen::Vector3d cur(pi.X(), pi.Y(), pi.Z());
        cum_len[static_cast<size_t>(i)] = cum_len[static_cast<size_t>(i - 1)] + (cur - prev).norm();
        prev = cur;
    }
    const double scale = (cum_len.back() > 1e-12) ? L / cum_len.back() : 1.0;
    for (double& c : cum_len)
        c *= scale;

    auto param_at_length = [&](double target) {
        if (target <= 0.0) return 0.0;
        if (target >= L) return 1.0;
        const auto it = std::lower_bound(cum_len.begin(), cum_len.end(), target);
        const int idx = static_cast<int>(std::distance(cum_len.begin(), it));
        if (idx <= 0) return 0.0;
        if (idx >= table_n) return 1.0;
        const double denom = cum_len[static_cast<size_t>(idx)] - cum_len[static_cast<size_t>(idx - 1)];
        const double t = (denom > 1e-12)
            ? (target - cum_len[static_cast<size_t>(idx - 1)]) / denom
            : 0.0;
        return params[static_cast<size_t>(idx - 1)]
            + t * (params[static_cast<size_t>(idx)] - params[static_cast<size_t>(idx - 1)]);
    };

    auto append_at = [&](double u) {
        const LNLib::XYZ p = LNLib::NurbsCurve::GetPointOnCurve(nurbs, u);
        out.emplace_back(Eigen::Vector3d(p.X(), p.Y(), p.Z()));
    };

    append_at(0.0);
    for (double d = ds; d < L; d += ds)
        append_at(param_at_length(d));
    if (out.empty() || (out.back() - Eigen::Vector3d(
            LNLib::NurbsCurve::GetPointOnCurve(nurbs, 1.0).X(),
            LNLib::NurbsCurve::GetPointOnCurve(nurbs, 1.0).Y(),
            LNLib::NurbsCurve::GetPointOnCurve(nurbs, 1.0).Z())).norm() > 1e-9)
        append_at(1.0);

    return out;
}

std::vector<Eigen::Vector3d> sample_line_by_ds(const Line3D& line, double ds)
{
    std::vector<Eigen::Vector3d> out;
    if (!line.valid) return out;

    const double L = (line.p1 - line.p0).norm();
    if (L <= 0.0 || ds <= 0.0) {
        out.push_back(line.p0);
        return out;
    }

    const int n = std::max(2, static_cast<int>(std::ceil(L / ds)) + 1);
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double alpha = static_cast<double>(i) / static_cast<double>(n - 1);
        out.emplace_back((1.0 - alpha) * line.p0 + alpha * line.p1);
    }
    return out;
}

void zhou_line_to_input_nurbs(Observation& v1, Observation& v2)
{
    const int pt_num = 5;
    auto fit_line = [pt_num](Observation& obs) {
        cv::Point2d p1 = obs.curve.points.front();
        cv::Point2d p2 = obs.curve.points.back();
        std::vector<LNLib::XYZ> pts;
        for (int i = 0; i < pt_num; ++i) {
            const double t = static_cast<double>(i) / (pt_num - 1);
            const cv::Point2d pt = p1 * (1.0 - t) + p2 * t;
            pts.emplace_back(LNLib::XYZ(pt.x, pt.y, 1.0));
        }
        LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, pts, recon_cfg().p_nurbs + 1, obs.curve.nurbs);
    };
    fit_line(v1);
    fit_line(v2);
}

bool fit_ordered_points_least_squares(const std::vector<Eigen::Vector3d>& fit_points,
                                      int ctrl_point_num,
                                      LNLib::LN_NurbsCurve& nurbs,
                                      int& ctrl_used)
{
    ctrl_used = 0;
    if (fit_points.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1)) return false;

    const int n = static_cast<int>(fit_points.size());
    ctrl_point_num = std::clamp(ctrl_point_num, recon_cfg().p_nurbs + 1, n);

    std::vector<LNLib::XYZ> points;
    points.reserve(fit_points.size());
    for (const auto& p : fit_points)
        points.emplace_back(LNLib::XYZ(p.x(), p.y(), p.z()));

    if (!LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, points, ctrl_point_num, nurbs))
        return false;

    ctrl_used = ctrl_point_num;
    return true;
}

void log_recon_common(int feature_id, int n_ctrl)
{
    std::cerr << std::fixed << std::setprecision(6)
              << "[recon] feature_id=" << feature_id
              << " P_NURBS=" << recon_cfg().p_nurbs
              << " W1=" << recon_cfg().w1
              << " n_ctrl=" << n_ctrl << std::endl;
}

void log_nurbs_fit_method(const char* tag, int feature_id, int n_sample,
                          double ds, bool is_ours, int n_rbf)
{
    std::cerr << std::fixed << std::setprecision(6)
              << "[recon][" << tag << "] feature_id=" << feature_id
              << " W3=" << recon_cfg().w3
              << " n_sample=" << n_sample
              << " ds=" << ds;
    if (is_ours)
        std::cerr << " | W2=" << recon_cfg().w2 << " n_RBF=" << n_rbf << " K_RBF=" << recon_cfg().k_rbf;
    std::cerr << std::endl;
}

void ensure_recon_common_logged(StereoFeature& f, int n_ctrl)
{
    if (f.recon_common_logged) return;
    log_recon_common(f.feature_id, n_ctrl);
    f.recon_common_logged = true;
}

Eigen::Matrix3d skew_sym(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

double rotation_angle_rad(const Eigen::Matrix3d& R)
{
    const double cos_theta = (R.trace() - 1.0) * 0.5;
    return std::acos(std::clamp(cos_theta, -1.0, 1.0));
}

Eigen::Vector3d random_unit_vector(std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Vector3d v(dist(rng), dist(rng), dist(rng));
    const double n = v.norm();
    if (n < 1e-12) return Eigen::Vector3d::UnitX();
    return v / n;
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& omega)
{
    const double theta = omega.norm();
    if (theta < 1e-12) return Eigen::Matrix3d::Identity();
    const Eigen::Vector3d axis = omega / theta;
    const Eigen::Matrix3d K = skew_sym(axis);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

}

void CurveReconstructor::write_nurbs_fit_spec(StereoFeature& f, int sample_num, int ctrl_point_num,
                                                int n_rbf, double ds)
{
    f.nurbs_fit_spec_.sample_num = sample_num;
    f.nurbs_fit_spec_.ctrl_point_num = ctrl_point_num;
    f.nurbs_fit_spec_.n_rbf = n_rbf;
    f.nurbs_fit_spec_.ds = ds;
    if (f.nurbs_fit_spec_.corr_delta_s <= 0.0)
        f.nurbs_fit_spec_.corr_delta_s = ds;
    f.nurbs_fit_spec_.valid = (sample_num > 5 && ctrl_point_num >= recon_cfg().p_nurbs + 1);
    ensure_recon_common_logged(f, ctrl_point_num);
}

CurveReconstructor::CurveReconstructor() = default;

void CurveReconstructor::fill_observation(Observation& obs,
                                            const CFeature& feat,
                                            const Camera& camera,
                                            const Eigen::Matrix3d& R,
                                            const Eigen::Vector3d& t)
{
    obs.curve = feat.curve;
    obs.camera = camera;
    obs.R = R;
    obs.t = t;
    obs.s_ids.clear();
    obs.s_values.clear();
    obs.s_points.clear();
    obs.angle_diffs.clear();
    obs.filled = true;
}

StereoFeature* CurveReconstructor::find_feature(int feature_id)
{
    auto it = std::find_if(features_.begin(), features_.end(),
                           [feature_id](const StereoFeature& sf) {
                               return sf.feature_id == feature_id;
                           });
    return it == features_.end() ? nullptr : &(*it);
}

bool CurveReconstructor::has_stereo_observation(const StereoFeature& f) const
{
    return f.view1.filled && f.view2.filled;
}

void CurveReconstructor::add_frame(int view_index,
                                     std::vector<CFeature>& new_features,
                                     const Eigen::Vector3d& gt_position,
                                     const Eigen::Quaterniond& gt_quaternion,
                                     const Camera& camera)
{
    assert(view_index == 0 || view_index == 1);

    if (view_index == 0) {
        features_.clear();
        feature_colors_.clear();
        denseMap_ = cv::Mat();
    } else if (features_.empty()) {
        std::cerr << "[error][add_frame] view1 received before view0" << std::endl;
        return;
    }

    image_width_ = camera.img_width + 600;
    image_height_ = camera.img_height + 600;

    const Eigen::Matrix3d Rs = gt_quaternion.toRotationMatrix();
    const Eigen::Vector3d t_wc = gt_position + Rs * camera.TIC;
    const Eigen::Matrix3d R_wc = Rs * camera.RIC;

    if (view_index == 0) {
        for (const auto& feat : new_features) {
            if (feat.feature_id < 0) continue;

            StereoFeature sf;
            sf.feature_id = feat.feature_id;
            sf.alls_cnt = 0;
            fill_observation(sf.view1, feat, camera, R_wc, t_wc);
            std::uniform_int_distribution<int> dist(0, 255);
            feature_colors_[feat.feature_id] = cv::Scalar(
                dist(rng_), dist(rng_), dist(rng_));
            features_.push_back(std::move(sf));
        }
    } else {
        for (const auto& feat : new_features) {
            if (feat.feature_id < 0) continue;

            StereoFeature* sf = find_feature(feat.feature_id);
            if (!sf) {
                std::cerr << "[warn][add_frame] view1 missing feature_id="
                          << feat.feature_id << std::endl;
                continue;
            }

            fill_observation(sf->view2, feat, camera, R_wc, t_wc);

            if (sf->view1.curve.is_curved != feat.curve.is_curved) {
                std::cerr << "[warn][add_frame] curve/line mismatch, feature_id="
                          << feat.feature_id << std::endl;
            }
        }
    }
}

void CurveReconstructor::add_frame(int view_index,
                                     std::vector<CFeature>& new_features,
                                     const Eigen::Vector3d& gt_position,
                                     const Eigen::Quaterniond& gt_quaternion,
                                     const Camera& camera,
                                     const cv::Mat& denseMap)
{
    add_frame(view_index, new_features, gt_position, gt_quaternion, camera);
    if (view_index == 1 && !denseMap.empty())
        denseMap_ = denseMap.clone();
}

void CurveReconstructor::apply_pose_error()
{
    const auto& cfg = recon_cfg();
    if (cfg.trans_error_m <= 0.0 && cfg.rot_error_deg <= 0.0) return;

    const StereoFeature* ref = nullptr;
    for (const auto& f : features_) {
        if (has_stereo_observation(f)) {
            ref = &f;
            break;
        }
    }
    if (!ref) return;

    const Observation& v1 = ref->view1;
    const Observation& v2 = ref->view2;

    const Eigen::Matrix3d R1 = v1.R;
    const Eigen::Matrix3d R2 = v2.R;
    const Eigen::Vector3d t1 = v1.t;
    const Eigen::Vector3d t2 = v2.t;

    Eigen::Matrix3d R_gt = R1.transpose() * R2;
    Eigen::Vector3d t_gt = R1.transpose() * (t2 - t1);

    std::mt19937 rng(static_cast<uint32_t>(cfg.pose_error_seed));

    Eigen::Matrix3d R_err = R_gt;
    Eigen::Vector3d t_err = t_gt;

    double trans_mag = 0.0;
    if (cfg.trans_error_m > 0.0) {
        const Eigen::Vector3d u = random_unit_vector(rng);
        trans_mag = cfg.trans_error_m;
        t_err += trans_mag * u;
    }

    double rot_delta_deg = 0.0;
    if (cfg.rot_error_deg > 0.0) {
        const Eigen::Vector3d axis = random_unit_vector(rng);
        const double delta_theta = cfg.rot_error_deg * M_PI / 180.0;
        rot_delta_deg = cfg.rot_error_deg;
        R_err = exp_so3(delta_theta * axis) * R_gt;
    }

    const Eigen::Matrix3d R2_new = R1 * R_err;
    const Eigen::Vector3d t2_new = t1 + R1 * t_err;

    for (auto& f : features_) {
        if (!has_stereo_observation(f)) continue;
        f.view2.R = R2_new;
        f.view2.t = t2_new;
    }

    std::cerr << std::fixed << std::setprecision(6)
              << "[pose_error] baseline=" << t_gt.norm()
              << " trans_err=" << trans_mag
              << " rot_err=" << rot_delta_deg << "deg"
              << " (trans_m=" << cfg.trans_error_m
              << " rot_deg=" << cfg.rot_error_deg
              << " seed=" << cfg.pose_error_seed << ")"
              << std::endl;
}

bool CurveReconstructor::process()
{
    for (auto& f : features_) {
        if (!has_stereo_observation(f)) continue;

        auto& rf = f.resulted_feature;
        reset_resulted_feature_timing(rf, is_curve(f));

        if (!rf.is_curve) {
            TicToc tt;

            if (!reconstruct_line_dense(f)) {
                std::cerr << "[debug][Ours][process] feature_id=" << f.feature_id
                          << ": failed at reconstruct_line_dense" << std::endl;
                continue;
            }
            rf.time_consumptions.push_back(tt.toc());
        } else {
            TicToc tt;
            if (!compute_intra_curve_correspondence(f)) {
                std::cerr << "[debug][Ours][process] feature_id=" << f.feature_id
                          << ": failed at curve_correspondence" << std::endl;
                continue;
            }
            rf.time_consumptions.push_back(tt.toc());

            tt.tic();
            int sample_num = 0;
            double s1_span = 0.0, s2_span = 0.0;
            if (!approximate_depth_function(f, sample_num, s1_span, s2_span)) {
                std::cerr << "[debug][Ours][process] feature_id=" << f.feature_id
                          << ": failed at approximate_depth (stages="
                          << rf.time_consumptions.size() << ")" << std::endl;
                continue;
            }
            rf.time_consumptions.push_back(tt.toc());

            tt.tic();
            if (!post_processing(f, sample_num, s1_span, s2_span)) {
                std::cerr << "[debug][Ours][process] feature_id=" << f.feature_id
                          << ": failed at post_processing (stages="
                          << rf.time_consumptions.size() << ")" << std::endl;
                continue;
            }
            rf.time_consumptions.push_back(tt.toc());
        }

        finalize_overall_time(rf);

        if (rf.is_curve && rf.time_consumptions.size() >= 3) {
            std::cerr << std::fixed << std::setprecision(2)
                      << "[recon][Ours] feature_id=" << f.feature_id
                      << " corr=" << rf.time_consumptions[0] << "ms"
                      << " depth=" << rf.time_consumptions[1] << "ms"
                      << " post=" << rf.time_consumptions[2] << "ms"
                      << " total=" << rf.overall_time_consumption << "ms"
                      << std::endl;
        } else if (!rf.is_curve && !rf.time_consumptions.empty()) {
            std::cerr << std::fixed << std::setprecision(2)
                      << "[recon][Ours] feature_id=" << f.feature_id
                      << " line=" << rf.time_consumptions[0] << "ms"
                      << " total=" << rf.overall_time_consumption << "ms"
                      << std::endl;
        }
    }

    return true;
}

bool CurveReconstructor::process_Zhou()
{
    for (auto& f : features_) {
        if (!has_stereo_observation(f)) continue;

        auto& zrf = f.Zhou_resulted_feature;
        reset_resulted_feature_timing(zrf, is_curve(f));
        if (!Zhou_reconstruct(f)) {
            std::cerr << "[debug][Zhou] feature_id=" << f.feature_id
                      << ": reconstruction failed" << std::endl;
        }
    }
    return true;
}

bool CurveReconstructor::process_DLT()
{
    for (auto& f : features_) {
        if (!has_stereo_observation(f)) continue;

        auto& drf = f.DLT_resulted_feature;
        reset_resulted_feature_timing(drf, is_curve(f));

        TicToc tt;
        if (!DLT_reconstruct(f)) {
            std::cerr << "[debug][DLT] feature_id=" << f.feature_id
                      << ": reconstruction failed" << std::endl;
            continue;
        }
        drf.time_consumptions.push_back(tt.toc());

        finalize_overall_time(drf);
    }
    return true;
}

void CurveReconstructor::save_results(const std::string& save_folder, double ds)
{
    if (features_.empty()) {
        std::cerr << "[warn][save] features empty, nothing to save." << std::endl;
        return;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBL>::Ptr> raw_clouds(3);
    std::vector<pcl::PointCloud<pcl::PointXYZRGBL>::Ptr> curve_clouds(3);
    for (int i = 0; i < 3; ++i) {
        raw_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZRGBL>);
        curve_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZRGBL>);
    }

    auto append_points = [](const std::vector<Eigen::Vector3d>& pts,
                            pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud,
                            int feature_id, uint8_t r, uint8_t g, uint8_t b)
    {
        for (const auto& point : pts) {
            pcl::PointXYZRGBL pcl_point;
            pcl_point.x = point.x();
            pcl_point.y = point.y();
            pcl_point.z = point.z();
            pcl_point.r = r;
            pcl_point.g = g;
            pcl_point.b = b;
            pcl_point.label = feature_id;
            cloud->points.push_back(pcl_point);
        }
    };

    int features_with_two_frames = 0;
    for (const auto& f : features_) {
        if (!has_stereo_observation(f)) continue;
        ++features_with_two_frames;

        const int fid = f.feature_id;
        auto color_it = feature_colors_.find(fid);
        if (color_it == feature_colors_.end()) continue;
        cv::Scalar color = color_it->second;
        uint8_t b = color[0], g = color[1], r = color[2];

        const std::array<const ResultedFeature*, 3> results = {
            &f.resulted_feature, &f.Zhou_resulted_feature, &f.DLT_resulted_feature
        };

        for (int i = 0; i < 3; ++i) {
            const ResultedFeature& rf = *results[static_cast<size_t>(i)];
            if (!rf.fit_points.empty())
                append_points(rf.fit_points, raw_clouds[i], fid, r, g, b);

            std::vector<Eigen::Vector3d> curve_pts;
            if (rf.geometry == ResultGeometry::Nurbs && rf.has_nurbs) {
                const double nurbs_L = LNLib::NurbsCurve::ApproximateLength(
                    rf.nurbs, LNLib::IntegratorType::GaussLegendre);
                if (i == 1 && nurbs_L > k_max_zhou_nurbs_arc_length) {
                    std::cerr << std::fixed << std::setprecision(4)
                              << "[warn][save] feature_id=" << fid
                              << " method=Zhou: arc length L=" << nurbs_L
                              << " > " << k_max_zhou_nurbs_arc_length
                              << ", skip curve sampling (raw only)" << std::endl;
                } else {
                    curve_pts = sample_nurbs_by_ds(rf.nurbs, ds);
                }
            } else if (rf.geometry == ResultGeometry::Line && rf.line.valid) {
                curve_pts = sample_line_by_ds(rf.line, ds);
            }

            if (!curve_pts.empty())
                append_points(curve_pts, curve_clouds[i], fid, r, g, b);
        }
    }

    if (features_with_two_frames == 0) {
        std::cerr << "[warn][save] no feature with 2 frames, nothing to save." << std::endl;
        return;
    }

    const std::array<std::string, 3> raw_names = {
        "ours_raw_points.ply", "zhou_raw_points.ply", "dlt_raw_points.ply"
    };
    const std::array<std::string, 3> curve_names = {
        "ours_curve_points.ply", "zhou_curve_points.ply", "dlt_curve_points.ply"
    };

    int saved_count = 0;
    pcl::PLYWriter writer;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            const auto& cloud = (j == 0) ? raw_clouds[i] : curve_clouds[i];
            if (cloud->empty()) continue;

            cloud->width = cloud->points.size();
            cloud->height = 1;
            cloud->is_dense = false;

            const std::string& name = (j == 0) ? raw_names[i] : curve_names[i];
            writer.write(save_folder + name, *cloud, true);
            ++saved_count;
        }
    }

    if (saved_count == 0)
        std::cerr << "[warn][save] no PLY files written." << std::endl;
    else
        std::cout << std::fixed << std::setprecision(6)
                  << "Reconstruction results saved to " << save_folder
                  << " (" << saved_count << " files, ds=" << ds << ")" << std::endl;
}

void CurveReconstructor::get_runtimes(RuntimeStats& stats) const
{
    stats = RuntimeStats{};

    std::vector<double> curve_corr;
    std::vector<double> curve_depth;
    std::vector<double> curve_post;
    std::vector<double> curve_total;
    std::vector<double> line_total;
    std::vector<double> ours_all;
    std::vector<double> zhou_raw;
    std::vector<double> zhou_nurbs;
    std::vector<double> dlt_total;

    for (const auto& f : features_) {
        if (!has_stereo_observation(f)) continue;

        const auto& ours = f.resulted_feature;
        if (ours.is_done) {
            ours_all.push_back(ours.overall_time_consumption);
            if (ours.is_curve) {
                if (ours.time_consumptions.size() >= 3) {
                    curve_corr.push_back(ours.time_consumptions[0]);
                    curve_depth.push_back(ours.time_consumptions[1]);
                    curve_post.push_back(ours.time_consumptions[2]);
                }
                curve_total.push_back(ours.overall_time_consumption);
            } else {
                line_total.push_back(ours.overall_time_consumption);
            }
        }

        const auto& zhou = f.Zhou_resulted_feature;
        if (zhou.is_done && zhou.time_consumptions.size() >= 2) {
            zhou_raw.push_back(zhou.time_consumptions[0]);
            zhou_nurbs.push_back(zhou.time_consumptions[1]);
        }

        const auto& dlt = f.DLT_resulted_feature;
        if (dlt.is_done)
            dlt_total.push_back(dlt.overall_time_consumption);
    }

    stats.ours_curve_corr = average_or_placeholder(curve_corr);
    stats.ours_curve_depth = average_or_placeholder(curve_depth);
    stats.ours_curve_post = average_or_placeholder(curve_post);
    stats.ours_curve_total = average_or_placeholder(curve_total);
    stats.ours_line_total = average_or_placeholder(line_total);
    stats.ours_all_total = average_or_placeholder(ours_all);
    stats.zhou_raw = average_or_placeholder(zhou_raw);
    stats.zhou_nurbs = average_or_placeholder(zhou_nurbs);
    if (stats.zhou_raw == RuntimeStats::kPlaceholder
        || stats.zhou_nurbs == RuntimeStats::kPlaceholder) {
        stats.zhou_total = RuntimeStats::kPlaceholder;
    } else {
        stats.zhou_total = stats.zhou_raw + stats.zhou_nurbs;
    }
    stats.dlt_total = average_or_placeholder(dlt_total);
}

bool CurveReconstructor::is_curve(const StereoFeature& f) const
{
    return f.view1.curve.is_curved;
}

bool CurveReconstructor::reconstruct_line_dense(StereoFeature& f)
{
    const Observation& v1 = f.view1;
    const Observation& v2 = f.view2;

    if (v1.curve.is_curved != v2.curve.is_curved) return false;
    if (v1.curve.is_curved) return false;
    if (denseMap_.empty()) return false;
    if (v1.curve.pixels.empty()) return false;

    std::vector<cv::Point2f> pts1;
    pts1.reserve(v1.curve.pixels.size());
    for (const auto& pixel : v1.curve.pixels)
        pts1.emplace_back(cv::Point2f(pixel));

    std::vector<cv::Point2f> pts2;
    warp_points(pts1, denseMap_, pts2);
    if (pts2.size() != pts1.size()) return false;

    const Eigen::Matrix3d R1 = v1.R;
    const Eigen::Matrix3d R2 = v2.R;
    const Eigen::Vector3d t1 = v1.t;
    const Eigen::Vector3d t2 = v2.t;
    const Eigen::Matrix3d K1 = v1.camera.K;
    const Eigen::Matrix3d K2 = v2.camera.K;

    std::vector<Eigen::Vector3d> pts_3d;
    DLT_points(pts1, pts2, R1, t1, R2, t2, K1, K2, pts_3d);
    pts_3d.erase(std::remove_if(pts_3d.begin(), pts_3d.end(),
        [](const Eigen::Vector3d& p) {
            return !std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z());
        }), pts_3d.end());
    if (pts_3d.size() < 20) return false;

    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (const auto& p : pts_3d) c += p;
    c /= static_cast<double>(pts_3d.size());

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (const auto& p : pts_3d) {
        const Eigen::Vector3d d = p - c;
        M += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
    if (es.info() != Eigen::Success) return false;
    Eigen::Vector3d dir = es.eigenvectors().col(2);
    dir.normalize();

    double tmin = std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts_3d) {
        const double t = (p - c).dot(dir);
        tmin = std::min(tmin, t);
        tmax = std::max(tmax, t);
    }
    const Eigen::Vector3d p0 = c + tmin * dir;
    const Eigen::Vector3d p1 = c + tmax * dir;

    const double line_length = (p1 - p0).norm();
    if (!std::isfinite(line_length) || line_length > 10.0) {
        std::cerr << "[debug][Ours] feature_id=" << f.feature_id
                  << ": line length invalid (length=" << line_length << ")"
                  << std::endl;
        return false;
    }

    auto& rf = f.resulted_feature;
    rf.fit_points = std::move(pts_3d);
    rf.line.p0 = p0;
    rf.line.p1 = p1;
    rf.line.valid = true;
    rf.geometry = ResultGeometry::Line;
    rf.has_nurbs = false;
    rf.is_done = true;
    return true;
}

bool CurveReconstructor::compute_intra_curve_correspondence(StereoFeature& f)
{
    Observation& v1 = f.view1;
    Observation& v2 = f.view2;

    if (v1.curve.is_curved != v2.curve.is_curved) return false;
    if (!v1.curve.is_curved) return false;
    if (denseMap_.empty()) return false;

    const double S = std::sqrt(static_cast<double>(image_height_) * image_width_);
    const double scale = S / recon_cfg().s0;
    const double delta_s = recon_cfg().w3 / v1.curve.arc_length * scale;
    const double dis_thre = recon_cfg().match_dis_thre * scale;
    const int cur_pt_num = static_cast<int>(std::round(1.0 / delta_s));
    f.nurbs_fit_spec_.corr_delta_s = delta_s;

    v1.s_ids.clear();
    v1.s_values.clear();
    v1.s_points.clear();
    v1.s_ids.reserve(cur_pt_num);
    v1.s_values.reserve(cur_pt_num);
    v1.s_points.reserve(cur_pt_num);
    for (int i = 0; i < cur_pt_num; ++i) {
        const int s_id = f.alls_cnt++;
        v1.s_ids.push_back(s_id);

        const double s = static_cast<double>(i) / static_cast<double>(cur_pt_num - 1);
        v1.s_values.push_back(s);

        const LNLib::XYZ pt = LNLib::NurbsCurve::GetPointOnCurve(v1.curve.nurbs, s);
        v1.s_points.emplace_back(cv::Point2f(pt.X(), pt.Y()));
    }
    v1.angle_diffs = std::vector<double>(cur_pt_num, 0.0);

    assert(!v1.s_points.empty());

    const Eigen::Matrix3d R1 = v1.R;
    const Eigen::Matrix3d R2 = v2.R;
    const Eigen::Vector3d t1 = v1.t;
    const Eigen::Vector3d t2 = v2.t;
    const Eigen::Matrix3d R = R1.transpose() * R2;
    const Eigen::Vector3d t = R1.transpose() * (t2 - t1);
    const Eigen::Matrix3d E = skew(t) * R;
    const Eigen::Matrix3d F = v1.camera.K.inverse().transpose() * E * v2.camera.K.inverse();

    std::vector<cv::Point2f> matched_pts;
    warp_points(v1.s_points, denseMap_, matched_pts);

    v2.s_values.clear();
    v2.s_ids.clear();
    v2.s_points.clear();
    v2.angle_diffs.clear();

    for (size_t i = 0; i < v1.s_points.size(); ++i) {
        const Eigen::Vector3d x1(v1.s_points[i].x, v1.s_points[i].y, 1.0);
        const Eigen::Vector3d epipolar_line2 = F.transpose() * x1;

        double s_warm = std::numeric_limits<double>::quiet_NaN();
        if (!v2.s_values.empty() && i > 0) {
            s_warm = v2.s_values.back()
                   + (v1.s_values[i] - v1.s_values[i - 1]);
            s_warm = std::clamp(s_warm, 0.0, 1.0);
        }
        EpipolarCorrespondenceResult match;
        if (!find_epipolar_correspondence_from_warp(
                v2.curve.nurbs,
                epipolar_line2.x(), epipolar_line2.y(), epipolar_line2.z(),
                matched_pts[i], s_warm, match))
            continue;
        const double dis_min = match.warp_distance;
        const double s_nearest = match.s;
        const cv::Point2f point_nearest = match.xy;

        if (dis_min > dis_thre) {
            if (v2.s_points.empty()) continue;

            const double dis_prev = cv::norm(v2.s_points.back() - point_nearest);
            if (dis_prev > dis_thre) continue;
        }

        v2.s_values.push_back(s_nearest);
        v2.s_ids.push_back(v1.s_ids[i]);
        v2.s_points.emplace_back(point_nearest);

        const std::vector<LNLib::XYZ> derivs2 =
            LNLib::NurbsCurve::ComputeRationalCurveDerivatives(v2.curve.nurbs, 1, s_nearest);
        const Eigen::Vector2d tau2(derivs2[1].X(), derivs2[1].Y());
        const double a = -tau2[1], b = tau2[0], c = -(a * point_nearest.x + b * point_nearest.y);
        const Eigen::Vector3d tangent_line2(a, b, c);

        const double angle_dif = angle_difference(epipolar_line2, tangent_line2);
        v2.angle_diffs.emplace_back(angle_dif);
    }

    return true;
}

bool CurveReconstructor::approximate_depth_function(StereoFeature& f,
                                                      int& sample_num,
                                                      double& s1_span,
                                                      double& s2_span)
{
    const int fid = f.feature_id;
    Observation& v1 = f.view1;
    Observation& v2 = f.view2;

    if (v1.curve.is_curved != v2.curve.is_curved) return false;
    if (!v1.curve.is_curved) return false;
    if (v1.s_ids.empty() || v2.s_ids.empty()) {
        std::cerr << "[debug][Ours] feature_id=" << fid
                  << ": skip empty s_ids (it1=" << v1.s_ids.size()
                  << ", it2=" << v2.s_ids.size() << ")" << std::endl;
        return false;
    }

    const Eigen::Matrix3d R1 = v1.R;
    const Eigen::Matrix3d R2 = v2.R;
    const Eigen::Vector3d t1 = v1.t;
    const Eigen::Vector3d t2 = v2.t;

    TicToc tt_depth;

    std::vector<double> s1, s2, w1, w2;
    std::vector<Eigen::Vector3d> p1, p2;
    for (size_t i = 0; i < v2.s_ids.size(); ++i) {
        const int s_id = v2.s_ids[i];
        const auto s_it = std::find_if(v1.s_ids.begin(), v1.s_ids.end(),
                                       [s_id](int id) { return id == s_id; });
        if (s_it == v1.s_ids.end()) continue;

        const int idx1 = static_cast<int>(std::distance(v1.s_ids.begin(), s_it));
        const int idx2 = static_cast<int>(i);

        w1.push_back(v1.angle_diffs[idx1]);
        w2.push_back(v2.angle_diffs[idx2]);

        s1.push_back(v1.s_values[idx1]);
        s2.push_back(v2.s_values[idx2]);
        const Eigen::Vector2d p1_2d(v1.s_points[idx1].x, v1.s_points[idx1].y);
        const Eigen::Vector2d p2_2d(v2.s_points[idx2].x, v2.s_points[idx2].y);

        Eigen::Vector3d p1_3d, p2_3d;
        v1.camera.liftProjective(p1_2d, p1_3d);
        v2.camera.liftProjective(p2_2d, p2_3d);
        p1.emplace_back(p1_3d);
        p2.emplace_back(p2_3d);
    }
    if (s1.size() < 10) {
        std::cerr << "[debug][Ours] feature_id=" << fid
                  << ": too few correspondences for RBF (n=" << s1.size() << ", need>=10)"
                  << std::endl;
        return false;
    }

    const double S = std::sqrt(static_cast<double>(image_height_) * image_width_);
    const int _temp = static_cast<int>(std::round(
        0.5 * (v1.curve.arc_length + v2.curve.arc_length) * recon_cfg().w2 / S));
    const int num_basis_functions = std::max(3, _temp) + 1;

    Eigen::VectorXd coeffs1, coeffs2;
    std::vector<double> centers1, centers2, sigmas1, sigmas2;
    double residual = 0.0;
    TicToc tt_rbf;
    approximate_depth_with_GaussianRBF(p1, p2, s1, s2, w1, w2, R1, t1, R2, t2, num_basis_functions,
                                       coeffs1, coeffs2, centers1, centers2, sigmas1, sigmas2, residual);

    const double s1_min = *std::min_element(s1.begin(), s1.end());
    const double s1_max = *std::max_element(s1.begin(), s1.end());
    const double s2_min = *std::min_element(s2.begin(), s2.end());
    const double s2_max = *std::max_element(s2.begin(), s2.end());
    if ((s1_max - s1_min) < 0.1 || (s2_max - s2_min) < 0.1) {
        std::cerr << "[debug][Ours] feature_id=" << fid
                  << ": s-parameter span too short (s1_span="
                  << (s1_max - s1_min) << ", s2_span=" << (s2_max - s2_min) << ")"
                  << std::endl;
        return false;
    }

    const double _rough_2Dlength = v1.curve.arc_length * (s1_max - s1_min);
    const double _rough_z = get_approximated_depth(centers1, sigmas1, coeffs1, 0.0);
    if (_rough_z <= 0 || _rough_z > 500) {
        std::cerr << std::fixed << std::setprecision(4)
                  << "[debug][Ours] feature_id=" << fid
                  << ": rough depth=" << _rough_z
                  << ", RBF_residual=" << residual
                  << ", n_corr=" << s1.size()
                  << ", s1_span=" << (s1_max - s1_min)
                  << ", s2_span=" << (s2_max - s2_min) << ")"
                  << std::endl;
        return false;
    }

    s1_span = s1_max - s1_min;
    s2_span = s2_max - s2_min;

    sample_num = static_cast<int>(s1.size());
    const int ctrl_from_v1 = static_cast<int>(std::round(v1.curve.nurbs.ControlPoints.size() * s1_span));
    const int ctrl_from_v2 = static_cast<int>(std::round(v2.curve.nurbs.ControlPoints.size() * s2_span));
    int ctrl_point_num = std::max(ctrl_from_v1, ctrl_from_v2);
    ctrl_point_num = std::clamp(ctrl_point_num, recon_cfg().p_nurbs + 1, static_cast<int>(std::round(0.8 * sample_num)));
    const double ds_corr = f.nurbs_fit_spec_.corr_delta_s;
    write_nurbs_fit_spec(f, sample_num, ctrl_point_num, num_basis_functions, ds_corr);

    if (sample_num <= 5) {
        std::cerr << "[debug][Ours] feature_id=" << fid
                  << ": too few s1_k for 3D fit (" << sample_num
                  << ", _rough_z=" << _rough_z
                  << ", _rough_2Dlength=" << _rough_2Dlength << ")"
                  << std::endl;
        return false;
    }

    std::vector<Eigen::Vector3d> curve_3d_1;
    const auto inf = std::numeric_limits<double>::infinity();
    double z1_min = inf, z1_max = -inf;
    bool depth_valid = true;
    size_t fail_k = 0;
    double fail_z1 = 0;
    curve_3d_1.reserve(s1.size());
    for (size_t k = 0; k < s1.size(); ++k) {
        const double s1_k = s1[k];
        const double _z1 = get_approximated_depth(centers1, sigmas1, coeffs1, s1_k);

        if (_z1 <= 0) {
            depth_valid = false;
            fail_k = k;
            fail_z1 = _z1;
            break;
        }
        if (z1_min > _z1) z1_min = _z1;
        if (z1_max < _z1) z1_max = _z1;

        const LNLib::XYZ _uv1 = LNLib::NurbsCurve::GetPointOnCurve(v1.curve.nurbs, s1_k);
        Eigen::Vector3d _p1_cam;
        v1.camera.liftProjective(Eigen::Vector2d(_uv1.X(), _uv1.Y()), _p1_cam);
        _p1_cam *= _z1;
        const Eigen::Vector3d _p1 = t1 + R1 * _p1_cam;
        curve_3d_1.emplace_back(_p1);
    }

    if (!depth_valid) {
        std::cerr << std::fixed << std::setprecision(4)
                  << "[debug][Ours] feature_id=" << fid
                  << ": invalid depth at s1_k (non-positive z)"
                  << " at k=" << fail_k << "/" << sample_num
                  << ", z1=" << fail_z1
                  << ", _rough_z=" << _rough_z
                  << ", RBF_residual=" << residual
                  << ", s1_span=" << s1_span << ")"
                  << std::endl;
        return false;
    }
    const double depth_thresh = 1.0;
    if ((z1_max - z1_min) > depth_thresh) return false;

    f.resulted_feature.fit_points = std::move(curve_3d_1);

    return true;
}

bool CurveReconstructor::post_processing(StereoFeature& f,
                                           int sample_num,
                                           double s1_span,
                                           double s2_span)
{
    const int fid = f.feature_id;
    const Observation& v1 = f.view1;
    const Observation& v2 = f.view2;
    auto& rf = f.resulted_feature;

    if (rf.fit_points.empty()) return false;

    const int ctrl_from_v1 = static_cast<int>(std::round(v1.curve.nurbs.ControlPoints.size() * s1_span));
    const int ctrl_from_v2 = static_cast<int>(std::round(v2.curve.nurbs.ControlPoints.size() * s2_span));
    int ctrl_point_num = std::max(ctrl_from_v1, ctrl_from_v2);
    ctrl_point_num = std::clamp(ctrl_point_num, recon_cfg().p_nurbs + 1, static_cast<int>(std::round(sample_num * 0.8)));

    LNLib::LN_NurbsCurve curve;
    int ctrl_used = 0;
    TicToc tt_nurbs;
    const bool ls_ok = fit_ordered_points_least_squares(
        rf.fit_points, ctrl_point_num, curve, ctrl_used);

    if (!ls_ok) {
        rf.has_nurbs = false;
        rf.geometry = ResultGeometry::Empty;
        return false;
    }

    Curve _curve;
    _curve.nurbs = curve;
    _curve.compute_arc_length();
    TicToc tt_curva;
    const int curva_samples = static_cast<int>(std::max(10.0, _curve.arc_length) * 2);
    _curve.compute_curvature(curva_samples);

    const double max_curva = *std::max_element(_curve.curvature.begin(), _curve.curvature.end());
    const double curva_thre = 1.0e3;
    if (max_curva > curva_thre) {
        std::cerr << "[debug][Ours] feature_id=" << fid
                  << ": curvature too large (max_curva=" << max_curva
                  << ", thre=" << curva_thre << ")" << std::endl;
        rf.has_nurbs = false;
        rf.geometry = ResultGeometry::Empty;
        return false;
    }

    rf.nurbs = curve;
    rf.has_nurbs = true;
    rf.geometry = ResultGeometry::Nurbs;
    rf.ctrl_point_num = ctrl_used;
    rf.is_done = true;
    log_nurbs_fit_method("Ours", fid, sample_num,
                         f.nurbs_fit_spec_.ds, true, f.nurbs_fit_spec_.n_rbf);

    return true;
}

bool CurveReconstructor::Zhou_reconstruct(StereoFeature& f)
{
    Observation& v1 = f.view1;
    Observation& v2 = f.view2;

    if (v1.curve.is_curved != v2.curve.is_curved) return false;

    auto& zrf = f.Zhou_resulted_feature;
    TicToc tt;

    if (!v1.curve.is_curved && !v2.curve.is_curved)
        zhou_line_to_input_nurbs(v1, v2);

    BaselineReconstructor baseline(v1.curve, v2.curve, v1.R, v2.R, v1.t, v2.t, v1.camera, v2.camera);

    if (f.nurbs_fit_spec_.valid && f.nurbs_fit_spec_.corr_delta_s > 0.0) {
        baseline.set_delta_s(f.nurbs_fit_spec_.corr_delta_s);
    } else if (f.nurbs_fit_spec_.valid && f.nurbs_fit_spec_.sample_num > 1) {

        baseline.set_s_grid_count(f.nurbs_fit_spec_.sample_num);
    } else {
        baseline.set_delta_s(0.01);
    }

    if (!baseline.process_up_to_optimization())
        return false;

    zrf.fit_points = baseline.optimized_points_ref();
    if (zrf.fit_points.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1))
        return false;

    zrf.time_consumptions.push_back(tt.toc());

    tt.tic();
    const int n = static_cast<int>(zrf.fit_points.size());
    int ctrl_point_num = recon_cfg().p_nurbs + 1;
    if (f.nurbs_fit_spec_.valid) {
        ctrl_point_num = std::clamp(f.nurbs_fit_spec_.ctrl_point_num, recon_cfg().p_nurbs + 1, n);
    } else {
        const int ctrl_from_v1 = static_cast<int>(v1.curve.nurbs.ControlPoints.size());
        const int ctrl_from_v2 = static_cast<int>(v2.curve.nurbs.ControlPoints.size());
        ctrl_point_num = std::clamp(std::max(ctrl_from_v1, ctrl_from_v2), recon_cfg().p_nurbs + 1, n);
    }

    LNLib::LN_NurbsCurve nurbs;
    int ctrl_used = 0;
    if (!fit_ordered_points_least_squares(zrf.fit_points, ctrl_point_num, nurbs, ctrl_used))
        return false;

    const double zhou_L = LNLib::NurbsCurve::ApproximateLength(
        nurbs, LNLib::IntegratorType::GaussLegendre);

    double ds_param = 0.0;
    int n_sample_log = n;
    if (f.nurbs_fit_spec_.valid && f.nurbs_fit_spec_.corr_delta_s > 0.0) {
        ds_param = f.nurbs_fit_spec_.corr_delta_s;
        n_sample_log = f.nurbs_fit_spec_.sample_num > 0 ? f.nurbs_fit_spec_.sample_num : n;
    } else if (f.nurbs_fit_spec_.valid && f.nurbs_fit_spec_.sample_num > 1) {
        n_sample_log = f.nurbs_fit_spec_.sample_num;
        ds_param = 1.0 / (f.nurbs_fit_spec_.sample_num - 1);
    } else {
        ds_param = 0.01;
    }
    ensure_recon_common_logged(f, ctrl_used);
    log_nurbs_fit_method("Zhou", f.feature_id, n_sample_log, ds_param, false, -1);

    zrf.time_consumptions.push_back(tt.toc());

    if (zhou_L > k_max_zhou_nurbs_arc_length) {
        std::cerr << std::fixed << std::setprecision(4)
                  << "[warn][Zhou] feature_id=" << f.feature_id
                  << ": NURBS arc length L=" << zhou_L
                  << " > " << k_max_zhou_nurbs_arc_length
                  << ", skip curve fit (raw only)" << std::endl;
        zrf.has_nurbs = false;
        zrf.geometry = ResultGeometry::Empty;
        zrf.ctrl_point_num = ctrl_used;
        zrf.is_done = true;
        finalize_overall_time(zrf);
        return true;
    }

    zrf.nurbs = nurbs;
    zrf.has_nurbs = true;
    zrf.geometry = ResultGeometry::Nurbs;
    zrf.ctrl_point_num = ctrl_used;
    zrf.is_done = true;
    finalize_overall_time(zrf);
    return true;
}

bool CurveReconstructor::DLT_reconstruct(StereoFeature& f)
{
    const Observation& v1 = f.view1;
    const Observation& v2 = f.view2;

    if (v1.curve.is_curved != v2.curve.is_curved) return false;
    if (denseMap_.empty()) return false;
    if (v1.curve.pixels.empty()) return false;

    std::vector<cv::Point2f> pts1;
    pts1.reserve(v1.curve.pixels.size());
    for (const auto& pixel : v1.curve.pixels)
        pts1.emplace_back(cv::Point2f(pixel));

    std::vector<cv::Point2f> pts2;
    warp_points(pts1, denseMap_, pts2);

    const Eigen::Matrix3d R1 = v1.R;
    const Eigen::Matrix3d R2 = v2.R;
    const Eigen::Vector3d t1 = v1.t;
    const Eigen::Vector3d t2 = v2.t;
    const Eigen::Matrix3d K1 = v1.camera.K;
    const Eigen::Matrix3d K2 = v2.camera.K;

    std::vector<Eigen::Vector3d> pts_3d_DLT;
    DLT_points(pts1, pts2, R1, t1, R2, t2, K1, K2, pts_3d_DLT);
    if (pts_3d_DLT.size() < static_cast<size_t>(recon_cfg().p_nurbs + 1))
        return false;

    auto& drf = f.DLT_resulted_feature;
    drf.fit_points = pts_3d_DLT;

    const int ctrl_override = f.nurbs_fit_spec_.valid ? f.nurbs_fit_spec_.ctrl_point_num : -1;
    LNLib::LN_NurbsCurve nurbs;
    int ctrl_used = 0;
    if (!fit_points_to_nurbs(pts_3d_DLT, nurbs, ctrl_used, v1.camera.img_height, ctrl_override))
        return false;

    const double dlt_L = LNLib::NurbsCurve::ApproximateLength(
        nurbs, LNLib::IntegratorType::GaussLegendre);
    if (dlt_L > k_max_zhou_nurbs_arc_length) {
        std::cerr << std::fixed << std::setprecision(4)
                  << "[debug][DLT] feature_id=" << f.feature_id
                  << ": NURBS arc length L=" << dlt_L
                  << " > " << k_max_zhou_nurbs_arc_length
                  << ", reconstruction rejected" << std::endl;
        return false;
    }

    const int n_dlt = static_cast<int>(pts_3d_DLT.size());
    const double ds_dlt = (n_dlt > 1) ? 1.0 / (n_dlt - 1) : 0.0;
    ensure_recon_common_logged(f, ctrl_used);
    log_nurbs_fit_method("DLT", f.feature_id, n_dlt, ds_dlt, false, -1);

    drf.nurbs = nurbs;
    drf.has_nurbs = true;
    drf.geometry = ResultGeometry::Nurbs;
    drf.ctrl_point_num = ctrl_used;
    drf.is_done = true;
    return true;
}
