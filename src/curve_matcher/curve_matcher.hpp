#ifndef CURVE_MATCHER_HPP
#define CURVE_MATCHER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <vector>
#include <string>
#include <memory>
#include "curve_detector.hpp"
#include "curve.hpp"
#include "camera.hpp"
#include "warp_points.hpp"

#include "LSD/precomp.hpp"

#define MAX_TF_NUM 1000

struct DetectionRuntimeStats {
    static constexpr double kPlaceholder = -1.0;

    double ll_angle = kPlaceholder;
    double region_grow = kPlaceholder;
    double fit_extract = kPlaceholder;
    double total = kPlaceholder;
};

struct FrameFeatures
{
    int frame_id;
    cv::Mat orig_img;
    cv::Mat color_img;
    cv::Mat img;

    std::vector<CFeature> features;

};
typedef std::shared_ptr<FrameFeatures> FrameFeaturesPtr;

class CurveMatcher
{
public:

    CurveMatcher();

    void process(const std::string& image_path, const Camera& camera);

    std::vector<CFeature> get_features() const;

    cv::Mat get_dense_map() const;

    bool get_matching_runtime(double& t) const;

    bool get_case_detection_stats(DetectionRuntimeStats& stats) const;

private:
    bool init_ = false;
    int frame_id_ = 0;
    int allfeature_cnt_ = 0;
    Camera camera_;
    cv::Mat denseMap_;

    void computeDenseMap();
    pybind11::object roma_module_;

    pybind11::object load_module();

    FrameFeaturesPtr curframe_, prevframe_;
    CurveDetector detector_;
    cv::Ptr<cv::LineSegmentDetector> lsd_;

    void detectFeatures(std::vector<CFeature>& detected_features, std::vector<double>& runtimes);

    void trackFeatures(std::vector<CFeature>& detected_features);

    std::vector<double> detection_runtimes_;
    double matching_runtime_ = 0.0;

    double case_det_ll_sum_ = 0.0;
    double case_det_region_sum_ = 0.0;
    double case_det_fit_sum_ = 0.0;
    int case_det_frame_count_ = 0;
    void accumulate_case_detection_stats();

};

#endif
