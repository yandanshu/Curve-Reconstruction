#include "find_s.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double eval_epipolar_residual(const LNLib::LN_NurbsCurve& nurbs,
                              double a, double b, double c, double s)
{
    s = std::clamp(s, 0.0, 1.0);
    const LNLib::XYZ p = LNLib::NurbsCurve::GetPointOnCurve(nurbs, s);
    return a * p.X() + b * p.Y() + c;
}

void curve_point_at(const LNLib::LN_NurbsCurve& nurbs, double s, double& x, double& y)
{
    s = std::clamp(s, 0.0, 1.0);
    const LNLib::XYZ p = LNLib::NurbsCurve::GetPointOnCurve(nurbs, s);
    x = p.X();
    y = p.Y();
}

double squared_dist_to_point(const LNLib::LN_NurbsCurve& nurbs,
                             const cv::Point2f& pt, double s)
{
    double x, y;
    curve_point_at(nurbs, s, x, y);
    const double dx = x - pt.x;
    const double dy = y - pt.y;
    return dx * dx + dy * dy;
}

double nearest_curve_param_to_point(const LNLib::LN_NurbsCurve& nurbs,
                                    const cv::Point2f& pt,
                                    double s_hint)
{
    constexpr int coarse_n = 16;
    double best_s = 0.0;
    double best_d = std::numeric_limits<double>::infinity();

    auto update = [&](double s) {
        const double d = squared_dist_to_point(nurbs, pt, s);
        if (d < best_d) {
            best_d = d;
            best_s = s;
        }
    };

    if (!std::isnan(s_hint) && s_hint >= 0.0 && s_hint <= 1.0) {
        const double window = 0.15;
        const double lo = std::max(0.0, s_hint - window);
        const double hi = std::min(1.0, s_hint + window);
        for (int i = 0; i < coarse_n; ++i)
            update(lo + (hi - lo) * i / (coarse_n - 1));
    } else {
        for (int i = 0; i < coarse_n; ++i)
            update(static_cast<double>(i) / (coarse_n - 1));
    }

    double lo = std::max(0.0, best_s - 0.05);
    double hi = std::min(1.0, best_s + 0.05);
    const double gr = 0.618033988749895;
    double x1 = hi - gr * (hi - lo);
    double x2 = lo + gr * (hi - lo);
    double f1 = squared_dist_to_point(nurbs, pt, x1);
    double f2 = squared_dist_to_point(nurbs, pt, x2);
    for (int i = 0; i < 20; ++i) {
        if (f1 > f2) {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + gr * (hi - lo);
            f2 = squared_dist_to_point(nurbs, pt, x2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - gr * (hi - lo);
            f1 = squared_dist_to_point(nurbs, pt, x1);
        }
    }
    return std::clamp(0.5 * (lo + hi), 0.0, 1.0);
}

bool newton_epipolar_numeric(const LNLib::LN_NurbsCurve& nurbs,
                             double a, double b, double c,
                             double s_init,
                             const EpipolarCorrespondenceOptions& opts,
                             double& s_out, double& residual_out)
{
    double s = std::clamp(s_init, 0.0, 1.0);
    const double h = opts.fd_step;

    for (int iter = 0; iter < opts.max_newton_iter; ++iter) {
        const double f = eval_epipolar_residual(nurbs, a, b, c, s);
        if (std::abs(f) < opts.tol) {
            s_out = s;
            residual_out = std::abs(f);
            return true;
        }

        const double f_plus = eval_epipolar_residual(nurbs, a, b, c, s + h);
        const double f_minus = eval_epipolar_residual(nurbs, a, b, c, s - h);
        const double df = (f_plus - f_minus) / (2.0 * h);
        if (std::abs(df) < 1e-14)
            break;

        s = std::clamp(s - f / df, 0.0, 1.0);
    }

    const double f_final = eval_epipolar_residual(nurbs, a, b, c, s);
    s_out = s;
    residual_out = std::abs(f_final);
    return residual_out < opts.tol * 10.0;
}

bool sparse_epipolar_fallback(const LNLib::LN_NurbsCurve& nurbs,
                              double a, double b, double c,
                              const cv::Point2f& warp_hint,
                              const EpipolarCorrespondenceOptions& opts,
                              EpipolarCorrespondenceResult& result)
{
    const int n = std::max(4, opts.fallback_samples);
    std::vector<double> ss(static_cast<size_t>(n));
    std::vector<double> fs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ss[static_cast<size_t>(i)] = static_cast<double>(i) / (n - 1);
        fs[static_cast<size_t>(i)] = eval_epipolar_residual(nurbs, a, b, c, ss[static_cast<size_t>(i)]);
    }

    bool any = false;
    double best_warp = std::numeric_limits<double>::infinity();

    for (int i = 0; i < n - 1; ++i) {
        if (fs[static_cast<size_t>(i)] * fs[static_cast<size_t>(i + 1)] > 0)
            continue;

        double s_try, res;
        const double mid = 0.5 * (ss[static_cast<size_t>(i)] + ss[static_cast<size_t>(i + 1)]);
        if (!newton_epipolar_numeric(nurbs, a, b, c, mid, opts, s_try, res))
            continue;

        double x, y;
        curve_point_at(nurbs, s_try, x, y);
        const double warp_d = std::hypot(x - warp_hint.x, y - warp_hint.y);
        if (warp_d < best_warp) {
            best_warp = warp_d;
            result.s = s_try;
            result.xy = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
            result.epipolar_residual = res;
            result.warp_distance = warp_d;
            result.converged = true;
            result.used_fallback = true;
            any = true;
        }
    }
    return any;
}

bool try_epipolar_from_seed(const LNLib::LN_NurbsCurve& nurbs,
                            double a, double b, double c,
                            const cv::Point2f& warp_hint,
                            double s_seed,
                            const EpipolarCorrespondenceOptions& opts,
                            EpipolarCorrespondenceResult& result)
{
    double s_out, res;
    if (!newton_epipolar_numeric(nurbs, a, b, c, s_seed, opts, s_out, res))
        return false;

    double x, y;
    curve_point_at(nurbs, s_out, x, y);
    result.s = s_out;
    result.xy = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    result.epipolar_residual = res;
    result.warp_distance = std::hypot(x - warp_hint.x, y - warp_hint.y);
    result.converged = true;
    result.used_fallback = false;
    return true;
}

}

bool find_epipolar_correspondence_from_warp(
    const LNLib::LN_NurbsCurve& nurbs,
    double a, double b, double c,
    const cv::Point2f& warp_hint,
    double s_warm_start,
    EpipolarCorrespondenceResult& result,
    const EpipolarCorrespondenceOptions& opts)
{
    result = EpipolarCorrespondenceResult{};

    const double s_from_warp = nearest_curve_param_to_point(nurbs, warp_hint, s_warm_start);

    std::vector<double> seeds;
    seeds.push_back(s_from_warp);
    if (!std::isnan(s_warm_start) && s_warm_start >= 0.0 && s_warm_start <= 1.0
        && std::abs(s_warm_start - s_from_warp) > 1e-6)
        seeds.push_back(s_warm_start);

    for (double s_seed : seeds) {
        if (try_epipolar_from_seed(nurbs, a, b, c, warp_hint, s_seed, opts, result))
            return true;
    }

    if (opts.enable_fallback)
        return sparse_epipolar_fallback(nurbs, a, b, c, warp_hint, opts, result);

    return false;
}
