#ifndef FIND_S_HPP
#define FIND_S_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "NurbsCurve.h"

struct EpipolarCorrespondenceResult {
    double s = 0.0;
    cv::Point2f xy;
    double epipolar_residual = 0.0;
    double warp_distance = 0.0;
    bool converged = false;
    bool used_fallback = false;
};

struct EpipolarCorrespondenceOptions {
    double tol = 1e-6;
    int max_newton_iter = 20;
    double fd_step = 1e-7;
    int fallback_samples = 20;
    bool enable_fallback = true;
};

bool find_epipolar_correspondence_from_warp(
    const LNLib::LN_NurbsCurve& nurbs,
    double a, double b, double c,
    const cv::Point2f& warp_hint,
    double s_warm_start,
    EpipolarCorrespondenceResult& result,
    const EpipolarCorrespondenceOptions& opts = {});

#endif
