#ifndef CURVE_HPP
#define CURVE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "alglib/interpolation.h"

#include "NurbsCurve.h"

class Curve
{
  public:
    bool is_curved;
    LNLib::LN_NurbsCurve nurbs;

    std::vector<cv::Point> pixels;
    std::vector<cv::Point2d> points;
    double arc_length;
    std::vector<double> curvature;

    void compute_arc_length();

    void compute_curvature(int num_samples = 1000);
};

struct CFeature
{

    int feature_id;

    Curve curve;
};

#endif
