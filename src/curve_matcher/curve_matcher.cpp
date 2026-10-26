#include "curve_matcher.hpp"
#include "tic_toc.h"
#include <stdexcept>

#ifndef CURVE_RECONSTRUCTION_ROOT
#define CURVE_RECONSTRUCTION_ROOT "."
#endif

CurveMatcher::CurveMatcher() : detector_(1, 1.0, 1.0, 2.0, ANGLE_TH, 1.0, 0.6, 1024),
                                             roma_module_(load_module())
{
    lsd_ = mycreateLineSegmentDetector(1, 1.0, 1.0, 2.0, 22.5, 1.0, 0.6, 1024);
}

pybind11::object CurveMatcher::load_module(){
    pybind11::module sys = pybind11::module::import("sys");
    pybind11::list sys_path = sys.attr("path");
    sys_path.append(std::string(CURVE_RECONSTRUCTION_ROOT) + "/RoMa_INFER/");
    return pybind11::module::import("RoMa_inference");
}

void CurveMatcher::computeDenseMap()
{

    cv::Mat img1, img2;
    cv::cvtColor(prevframe_->img, img1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curframe_->img, img2, cv::COLOR_GRAY2BGR);

    pybind11::array_t<uint8_t> np_img1(
        {img1.rows, img1.cols, img1.channels()},
        {static_cast<ssize_t>(img1.step[0]),
        static_cast<ssize_t>(img1.step[1]),
        static_cast<ssize_t>(1)},
        img1.data);
    pybind11::array_t<uint8_t> np_img2(
        {img2.rows, img2.cols, img2.channels()},
        {static_cast<ssize_t>(img2.step[0]),
        static_cast<ssize_t>(img2.step[1]),
        static_cast<ssize_t>(1)},
        img2.data);

    pybind11::object np_out = roma_module_.attr("inference")(np_img1, np_img2);

    auto result = np_out.cast<std::tuple<pybind11::array, pybind11::array, pybind11::array, pybind11::array>>();

    pybind11::array map12_np       = std::get<0>(result);
    pybind11::array map21_np       = std::get<1>(result);
    pybind11::array certainty12_np = std::get<2>(result);
    pybind11::array certainty21_np = std::get<3>(result);

    auto map12_np_c = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(map12_np);
    auto map21_np_c = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(map21_np);
    auto certainty12_np_c = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(certainty12_np);
    auto certainty21_np_c = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>(certainty21_np);

    cv::Mat map12(map12_np_c.shape(0), map12_np_c.shape(1), CV_32FC2, (void*)map12_np_c.mutable_data());
    cv::Mat map21(map21_np_c.shape(0), map21_np_c.shape(1), CV_32FC2, (void*)map21_np_c.mutable_data());
    cv::Mat certainty12(certainty12_np_c.shape(0), certainty12_np_c.shape(1), CV_32FC1, (void*)certainty12_np_c.mutable_data());
    cv::Mat certainty21(certainty21_np_c.shape(0), certainty21_np_c.shape(1), CV_32FC1, (void*)certainty21_np_c.mutable_data());

    map12 = map12.clone();
    map21 = map21.clone();
    certainty12 = certainty12.clone();
    certainty21 = certainty21.clone();

    int H = map12.rows;
    int W = map12.cols;

    cv::Mat map12_denormalized(H, W, CV_32FC2);
    map12.copyTo(map12_denormalized);

    cv::multiply(map12_denormalized,
                 cv::Scalar(0.5f*(W-1), 0.5f*(H-1)),
                 map12_denormalized);
    cv::add(map12_denormalized,
            cv::Scalar(0.5f*(W-1), 0.5f*(H-1)),
            map12_denormalized);

    denseMap_ = map12_denormalized.clone();

}

void CurveMatcher::detectFeatures(std::vector<CFeature>& detected_features, std::vector<double>& runtimes)
#if DETECTION_METHOD == 0
{
    std::vector<Curve> curves_new;
    runtimes.clear();
    TicToc tt;
    detector_.detect(curframe_->img, curves_new, runtimes);
    runtimes.push_back(tt.toc());

    for(int i = 0; i < curves_new.size(); ++i)
    {
        CFeature lc_feature;
        lc_feature.curve = curves_new[i];
        detected_features.push_back(lc_feature);
    }
}
#elif DETECTION_METHOD == 1
{
    std::vector<cv::Vec4f> lines;
    cv::Mat _runtimes;
    TicToc tt;
    lsd_->detect(curframe_->img, lines, _runtimes);
    double ll_angle_runtime = _runtimes.at<double>(0,0), region_grow_runtime = _runtimes.at<double>(0,1), fitting_runtime = _runtimes.at<double>(0,2);
    double overall_runtime = tt.toc();
    runtimes.clear();
    runtimes.push_back(ll_angle_runtime);
    runtimes.push_back(region_grow_runtime);
    runtimes.push_back(fitting_runtime);
    runtimes.push_back(overall_runtime);

    std::vector<cv::Vec4f> _lines;
    for(const auto& l : lines)
    {
        if(cv::norm(cv::Point2f(l[0], l[1]) - cv::Point2f(l[2], l[3])) < OBJ_MIN_LEN)
            continue;
        _lines.emplace_back(l);
    }
    lines = _lines;
    cv::Mat img_show = curframe_->color_img.clone();
    lsd_->drawSegments(img_show, lines);
#if VISUALIZATION
    cv::imshow("Detection by LSD", img_show);
    cv::waitKey(1);
#endif

    std::vector<Curve> curves_new;
    curves_new.reserve(lines.size());
    for(const auto& l : lines)
    {
        Curve obj;
        obj.is_curved = false;
        cv::Point2f p1(l[0], l[1]), p2(l[2], l[3]);
        double len = cv::norm(p1 - p2);
        int sam_num = static_cast<int>(std::round(len * 2));
        sam_num = std::max(sam_num, 2);
        for(int i = 0; i < sam_num; ++i)
        {
        double t = static_cast<double>(i) / (sam_num - 1);
        cv::Point2d p = p1 + t * (p2 - p1);
            obj.points.emplace_back(p);
        }

        curves_new.emplace_back(obj);
    }

    for(int i = 0; i < curves_new.size(); ++i)
    {
        CFeature lc_feature;
        lc_feature.curve = curves_new[i];
        detected_features.push_back(lc_feature);
    }
}
#endif

void CurveMatcher::trackFeatures(std::vector<CFeature>& detected_features)
{
    if(detected_features.empty()) return;
    if(prevframe_->features.empty()) return;
    computeDenseMap();

    std::vector<std::pair<int, int>> matched_pairs;
    std::vector<std::vector<cv::Point2f>> features, OF_features;
    for(const auto& feature : prevframe_->features)
    {

        std::vector<cv::Point2f> feature_float;
        for(const auto& pt : feature.curve.points)
        {
            feature_float.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }
        features.emplace_back(feature_float);

        std::vector<cv::Point2f> OF_feature;
        warp_points(feature_float, denseMap_, OF_feature);

        OF_features.emplace_back(OF_feature);
    }
    if(OF_features.empty())
    {
        std::cerr << "No features to be tracked!" << std::endl;
        return;
    }

    cv::Mat img1_bgr, img2_bgr;
    cv::cvtColor(prevframe_->img, img1_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curframe_->img, img2_bgr, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> points;
    std::vector<int> point_to_this_feature_map;
    for(size_t i = 0; i < detected_features.size(); ++i)
    {
        for(const auto& pt : detected_features[i].curve.points)
        {
            points.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
            point_to_this_feature_map.push_back(i);
        }
    }
    cv::Mat this_points = cv::Mat(points).reshape(1);
    cv::flann::Index flann_this(this_points, cv::flann::KDTreeIndexParams(1));

    std::vector<int> forward_match(OF_features.size(), -1);
    for (size_t i = 0; i < OF_features.size(); ++i) {
        std::unordered_map<int, int> vote_count;
        for (const auto& pt : OF_features[i]) {
            cv::Mat query = (cv::Mat_<float>(1, 2) << pt.x, pt.y);
            std::vector<int> indices(1);
            std::vector<float> dists(1);
            flann_this.knnSearch(query, indices, dists, 1, cv::flann::SearchParams());
            int matched_feature = point_to_this_feature_map[indices[0]];
            vote_count[matched_feature]++;
        }
        int max_votes = 0;
        int best_match = -1;
        for (const auto& [feature_id, count] : vote_count) {
            if (count > max_votes) {
                max_votes = count;
                best_match = feature_id;
            }
        }
        forward_match[i] = best_match;
    }

    points.clear();
    std::vector<int> point_to_last_feature_map;
    for(size_t i = 0; i < OF_features.size(); ++i)
    {
        for(const auto& pt : OF_features[i])
        {
            points.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
            point_to_last_feature_map.push_back(i);
        }
    }
    cv::Mat last_points = cv::Mat(points).reshape(1);
    cv::flann::Index flann_last(last_points, cv::flann::KDTreeIndexParams(1));

    std::vector<int> backward_match(detected_features.size(), -1);
    for (size_t i = 0; i < detected_features.size(); ++i) {
        std::unordered_map<int, int> vote_count;
        for (const auto& pt : detected_features[i].curve.points) {
            cv::Mat query = (cv::Mat_<float>(1, 2) << pt.x, pt.y);
            std::vector<int> indices(1);
            std::vector<float> dists(1);
            flann_last.knnSearch(query, indices, dists, 1, cv::flann::SearchParams());
            int matched_feature = point_to_last_feature_map[indices[0]];
            vote_count[matched_feature]++;
        }
        int max_votes = 0;
        int best_match = -1;
        for (const auto& [feature_id, count] : vote_count) {
            if (count > max_votes) {
                max_votes = count;
                best_match = feature_id;
            }
        }
        backward_match[i] = best_match;
    }

    for (size_t i = 0; i < forward_match.size(); ++i) {
        int j = forward_match[i];
        if (j >= 0 && j < static_cast<int>(backward_match.size()) && backward_match[j] == static_cast<int>(i)) {
            matched_pairs.emplace_back(i, j);
        }
    }

    for(const auto& pair : matched_pairs)
    {
        int idx1 = pair.first;
        int idx2 = pair.second;
        detected_features[idx2].feature_id = prevframe_->features[idx1].feature_id;
        curframe_->features.emplace_back(detected_features[idx2]);
    }

    int cur_f_num = matched_pairs.size();

    if(cur_f_num < MAX_TF_NUM)
    {
        for(size_t i = 0; i < detected_features.size(); ++i)
        {
            if(detected_features[i].feature_id == -1 && detected_features[i].curve.is_curved)
            {
                detected_features[i].feature_id = allfeature_cnt_++;
                curframe_->features.emplace_back(detected_features[i]);
                cur_f_num++;
                if(cur_f_num == MAX_TF_NUM) break;
            }
        }
    }
    if(cur_f_num < MAX_TF_NUM)
    {
        for(size_t i = 0; i < detected_features.size(); ++i)
        {
            if(detected_features[i].feature_id == -1 && !detected_features[i].curve.is_curved)
            {
                detected_features[i].feature_id = allfeature_cnt_++;
                curframe_->features.emplace_back(detected_features[i]);
                cur_f_num++;
                if(cur_f_num == MAX_TF_NUM) break;
            }
        }
    }
}

void CurveMatcher::process(const std::string& image_path, const Camera& camera)
{
    camera_ = camera;

    if(!init_){
        curframe_.reset(new FrameFeatures);
        prevframe_.reset(new FrameFeatures);
    }
    else{
        curframe_.reset(new FrameFeatures);
    }

    cv::Mat raw_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (raw_img.empty()) {
        throw std::runtime_error("Cannot read image: " + image_path);
    }
    curframe_->orig_img = raw_img.clone();

    int img_width = curframe_->orig_img.cols;
    int img_height = curframe_->orig_img.rows;
    cv::Mat undistortedImg(img_height + 600, img_width + 600, CV_8UC1, cv::Scalar(0));
    cv::remap(curframe_->orig_img, undistortedImg, camera_.undistMap1, camera_.undistMap2, cv::INTER_LINEAR);
    curframe_->img = undistortedImg.clone();
    cv::cvtColor(curframe_->img, curframe_->color_img, cv::COLOR_GRAY2BGR);
    curframe_->frame_id = frame_id_;

    std::vector<CFeature> detected_features;
    detectFeatures(detected_features, detection_runtimes_);
    accumulate_case_detection_stats();

    if(!init_)
    {
        for(auto& feature : detected_features) feature.feature_id = allfeature_cnt_++;
        curframe_->features = detected_features;
        prevframe_ = curframe_;
        frame_id_++;
        init_ = true;
        return;
    }
    else
        for(auto& feature : detected_features) feature.feature_id = -1;
    TicToc tt;
    trackFeatures(detected_features);
    matching_runtime_ = tt.toc();

    prevframe_ = curframe_;
    frame_id_++;
}

std::vector<CFeature> CurveMatcher::get_features() const
{
    return curframe_->features;
}

cv::Mat CurveMatcher::get_dense_map() const
{
    return denseMap_;
}

void CurveMatcher::accumulate_case_detection_stats()
{
    if (detection_runtimes_.size() < 3) return;
    case_det_ll_sum_ += detection_runtimes_[0];
    case_det_region_sum_ += detection_runtimes_[1];
    case_det_fit_sum_ += detection_runtimes_[2];
    case_det_frame_count_++;
}

bool CurveMatcher::get_case_detection_stats(DetectionRuntimeStats& stats) const
{
    if (case_det_frame_count_ <= 0) {
        stats = DetectionRuntimeStats{};
        return false;
    }
    const double n = static_cast<double>(case_det_frame_count_);
    stats.ll_angle = case_det_ll_sum_ / n;
    stats.region_grow = case_det_region_sum_ / n;
    stats.fit_extract = case_det_fit_sum_ / n;
    stats.total = stats.ll_angle + stats.region_grow + stats.fit_extract;
    return true;
}

bool CurveMatcher::get_matching_runtime(double& t) const
{
    t = matching_runtime_;
    return 1;
}
