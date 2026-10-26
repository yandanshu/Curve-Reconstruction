#ifndef CURVE_DETECTOR_HPP
#define CURVE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include <deque>
#include <unordered_set>
#include <Eigen/Dense>
#include <random>

#include "alglib/interpolation.h"
#include "alglib/ap.h"
#include "alglib/dataanalysis.h"

#include "curve.hpp"
#include "config.hpp"

using namespace cv;

# define CUR_PT_NUM  20
# define MAX_DF_NUM  1000
# define NOT_TEED    0

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "tic_toc.h"

#define M_3_2_PI    (3 * CV_PI) / 2
#define M_2__PI     (2 * CV_PI)

#ifndef M_LN10
#define M_LN10      2.30258509299404568402
#endif

#define NOTDEF      double(-1024.0)

#define NOTUSED     0
#define USED        1

#define RELATIVE_ERROR_FACTOR 100.0

const double DEG_TO_RADS = CV_PI / 180;

#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

inline double distSq(const double x1, const double y1,
                     const double x2, const double y2)
{
    return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

inline double dist(const double x1, const double y1,
                   const double x2, const double y2)
{
    return sqrt(distSq(x1, y1, x2, y2));
}

inline double angle_diff_signed(const double& a, const double& b)
{
    double diff = a - b;
    while(diff <= -CV_PI) diff += M_2__PI;
    while(diff >   CV_PI) diff -= M_2__PI;
    return diff;
}

inline double angle_diff(const double& a, const double& b)
{
    return std::fabs(angle_diff_signed(a, b));
}

inline bool double_equal(const double& a, const double& b)
{

    if(a == b) return true;

    double abs_diff = fabs(a - b);
    double aa = fabs(a);
    double bb = fabs(b);
    double abs_max = (aa > bb)? aa : bb;

    if(abs_max < DBL_MIN) abs_max = DBL_MIN;

    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

inline bool AsmallerB_YoverX(const cv::Point2d &a, const cv::Point2d &b) {
    if (a.y == b.y) return a.x < b.x;
    else return a.y < b.y;
}

inline double get_slope(cv::Point2d p1, cv::Point2d p2) {
    return ((int) ceil(p2.y) != (int) ceil(p1.y)) ? (p2.x - p1.x) / (p2.y - p1.y) : 0;
}

inline double get_limit(cv::Point2d p, int row, double slope) {
    return p.x + (row - p.y) * slope;
}

inline double log_gamma_windschitl(const double& x)
{
    return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log(x*sinh(1/x) + 1/(810.0*pow(x, 6.0)));
}

inline double log_gamma_lanczos(const double& x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
    double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double b = 0;
    for(int n = 0; n < 7; ++n)
    {
        a -= log(x + double(n));
        b += q[n] * pow(x, double(n));
    }
    return a + log(b);
}

struct PointCompare {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    }
};

class CurveDetector
{
public:

    CurveDetector(int _refine = LSD_REFINE_STD, double _scale = 0.8,
                         double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
                         double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

    void detect(const Mat& _image, std::vector<Curve>& _curves_new, std::vector<double>& runtimes);

    const double SCALE;

private:
    pybind11::object edge_det_module_;

    pybind11::object load_module();

    Mat gradient_image_;
    Mat angle_image_;

    Mat region_grow_image_;

    Mat image;
    Mat scaled_image;
    Mat_<double> angles;
    Mat_<double> modgrad;
    Mat_<uchar> used;

    Mat_<double> delta_angles;
    Mat skeletons_with_delta_angles;

    int img_width;
    int img_height;
    double LOG_NT;
    int MIN_REG_SIZE;

    const int doRefine;
    const double SIGMA_SCALE;
    const double QUANT;
    const double ANG_TH;
    const double LOG_EPS;
    const double DENSITY_TH;
    const int N_BINS;

    struct RegionPoint {
        int x;
        int y;
        uchar* used;
        double angle;
        double modgrad;

        bool operator==(const RegionPoint& other) const {
            return x == other.x && y == other.y;
        }
    };

    struct RegionPointHash {
        std::size_t operator()(const RegionPoint& p) const {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };

    struct normPoint
    {
        Point2i p;
        int norm;
    };

    std::vector<normPoint> ordered_points;

    struct rect
    {
        double x1, y1, x2, y2;
        double width;
        double x, y;
        double theta;
        double dx,dy;
        double prec;
        double p;

    };

    struct rect_end
    {
        double x, y;
        double theta;
        double width;
        double length;
        double p;
    };

    struct Object
    {
        std::deque<RegionPoint> region;
        double reg_angle;
        bool is_curve;
        Curve curve;
        rect rec;

    };

    CurveDetector& operator= (const CurveDetector&);

    void drawObjects_nurbs(const std::vector<Object>& objs, const std::string& img_name);

    void ll_angle(const double& threshold, const unsigned int& n_bins);

    bool get_region(const Point2i& s, std::deque<RegionPoint>& reg,
        const double& prec, const double p, double& reg_angle, std::deque<double>& local_angles);

    bool get_line(Object& line_obj, const Point2i& seed, const double prec, const double p);

    bool get_curve(Object& curve_obj);

    void region_grow(const Point2i& s, std::deque<RegionPoint>& reg,
                        double& reg_angle, const double& prec);

    void region2rect(const std::deque<RegionPoint>& reg, const double reg_angle,
                        const double prec, const double p, rect& rec) const;

    double get_theta(const std::deque<RegionPoint>& reg, const double& x,
                        const double& y, const double& reg_angle, const double& prec) const;

    bool refine(std::deque<RegionPoint>& reg, double reg_angle, const Point2i& s,
                const double prec, double p, rect& rec, const double& density_th);

    bool reduce_region_radius(std::deque<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th);

    double rect_improve(rect& rec) const;

    double rect_nfa(const rect& rec) const;

    double nfa(const int& n, const int& k, const double& p) const;

    bool isAligned(int x, int y, const double& theta, const double& prec) const;

public:

    static inline bool compare_norm( const normPoint& n1, const normPoint& n2 )
    {
        return (n1.norm > n2.norm);
    }

    struct PointHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    struct PointEqual {
        bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
            return a.first == b.first && a.second == b.second;
        }
    };

};

#endif
