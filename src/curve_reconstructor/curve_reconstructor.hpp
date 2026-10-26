#ifndef CURVE_RECONSTRUCTOR_HPP
#define CURVE_RECONSTRUCTOR_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <random>
#include <unordered_map>
#include <vector>
#include <string>

#include "curve.hpp"
#include "camera.hpp"
#include "NurbsCurve.h"

struct Line3D {
    Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
    bool valid = false;
};

enum class ResultGeometry { Empty, Nurbs, Line };

struct ResultedFeature {
    bool is_done = false;
    bool is_curve = false;

    ResultGeometry geometry = ResultGeometry::Empty;

    std::vector<Eigen::Vector3d> fit_points;

    LNLib::LN_NurbsCurve nurbs;
    bool has_nurbs = false;

    Line3D line;

    int ctrl_point_num = 0;

    std::vector<double> time_consumptions;
    double overall_time_consumption = 0.0;
};

struct NurbsFitSpec {
    bool valid = false;
    int sample_num = -1;
    int ctrl_point_num = -1;
    int n_rbf = -1;
    double ds = 0.0;
    double corr_delta_s = 0.0;
};

struct Observation {
    bool filled = false;
    Curve curve;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Camera camera;
    std::vector<int> s_ids;
    std::vector<double> s_values;
    std::vector<cv::Point2f> s_points;
    std::vector<double> angle_diffs;
};

struct StereoFeature {
    int feature_id = -1;
    int alls_cnt = 0;
    bool recon_common_logged = false;
    Observation view1;
    Observation view2;
    NurbsFitSpec nurbs_fit_spec_;
    ResultedFeature resulted_feature;
    ResultedFeature Zhou_resulted_feature;
    ResultedFeature DLT_resulted_feature;
};

struct RuntimeStats {
    static constexpr double kPlaceholder = -1.0;

    double ours_curve_corr = kPlaceholder;
    double ours_curve_depth = kPlaceholder;
    double ours_curve_post = kPlaceholder;
    double ours_curve_total = kPlaceholder;
    double ours_line_total = kPlaceholder;
    double ours_all_total = kPlaceholder;
    double zhou_raw = kPlaceholder;
    double zhou_nurbs = kPlaceholder;
    double zhou_total = kPlaceholder;
    double dlt_total = kPlaceholder;
};

class CurveReconstructor
{
public:
    CurveReconstructor();

    void add_frame(int view_index,
                   std::vector<CFeature>& new_features,
                   const Eigen::Vector3d& gt_position,
                   const Eigen::Quaterniond& gt_quaternion,
                   const Camera& camera);

    void add_frame(int view_index,
                   std::vector<CFeature>& new_features,
                   const Eigen::Vector3d& gt_position,
                   const Eigen::Quaterniond& gt_quaternion,
                   const Camera& camera,
                   const cv::Mat& denseMap);

    bool process();
    bool process_Zhou();
    bool process_DLT();

    void save_results(const std::string& save_folder, double ds = 0.005);
    void get_runtimes(RuntimeStats& stats) const;

    void apply_pose_error();

private:
    int image_width_ = 0;
    int image_height_ = 0;
    cv::Mat denseMap_;
    std::vector<StereoFeature> features_;
    std::unordered_map<int, cv::Scalar> feature_colors_;
    std::mt19937 rng_{std::random_device{}()};

    bool is_curve(const StereoFeature& f) const;

    bool reconstruct_line_dense(StereoFeature& f);

    bool compute_intra_curve_correspondence(StereoFeature& f);
    bool approximate_depth_function(StereoFeature& f,
                                    int& sample_num,
                                    double& s1_span,
                                    double& s2_span);
    bool post_processing(StereoFeature& f,
                         int sample_num,
                         double s1_span,
                         double s2_span);

    bool Zhou_reconstruct(StereoFeature& f);
    bool DLT_reconstruct(StereoFeature& f);

    static void fill_observation(Observation& obs,
                                 const CFeature& feat,
                                 const Camera& camera,
                                 const Eigen::Matrix3d& R,
                                 const Eigen::Vector3d& t);

    static void write_nurbs_fit_spec(StereoFeature& f, int sample_num, int ctrl_point_num,
                                     int n_rbf, double ds);

    StereoFeature* find_feature(int feature_id);
    bool has_stereo_observation(const StereoFeature& f) const;
};

#endif
