#include <opencv2/opencv.hpp>
#include "curve_detector.hpp"
#include "curve_matcher.hpp"
#include "curve_reconstructor.hpp"
#include "camera.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

#include <pybind11/embed.h>

#include "NurbsCurve.h"
#include "config.hpp"
#include "recon_config.hpp"

#include "tic_toc.h"

namespace fs = std::filesystem;

struct ImageData{
    int64_t timestamp;
    std::string filename;
    int camera_id;
};

struct GroundTruth{
    int64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond quaternion;
};

void loadGroundTruthAndImageList(const std::string& filepath, std::vector<GroundTruth>& gt_list, std::vector<ImageData>& image_list){

    gt_list.clear();
    image_list.clear();

    std::ifstream file(filepath);
    if (!file) throw std::runtime_error("Cannot open image list: " + filepath);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        GroundTruth gt;
        ImageData image_data;
        double px, py, pz, qw, qx, qy, qz;
        if (!(ss >> image_data.timestamp >> qw >> qx >> qy >> qz
                 >> px >> py >> pz >> image_data.camera_id >> image_data.filename)) {
            throw std::runtime_error("Invalid image record in " + filepath);
        }
        gt.timestamp = image_data.timestamp;
        // COLMAP world-to-camera pose to camera-to-world pose.
        gt.quaternion = Eigen::Quaterniond(qw, qx, qy, qz).conjugate();
        gt.position = -(gt.quaternion.toRotationMatrix() * Eigen::Vector3d(px, py, pz));
        gt_list.push_back(gt);
        image_list.push_back(image_data);
        std::getline(file, line);
    }
    if (image_list.size() != 2) {
        throw std::runtime_error("Expected two views in " + filepath);
    }
}

std::string resolve_recon_config_path(int argc, char** argv)
{
    std::string path = default_recon_config_path();
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) path = argv[++i];
        else if (arg.rfind("--config=", 0) == 0 && arg.size() > 9) path = arg.substr(9);
        else throw std::runtime_error("Expected --config PATH; got: " + arg);
    }
    return path;
}

int main(int argc, char** argv)
{
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: main [--config PATH]\n";
        return 0;
    }
    try {
        load_recon_config(resolve_recon_config_path(argc, argv));
        std::vector<fs::path> cases;
        for (const auto& entry : fs::directory_iterator(recon_cfg().dataset_root)) {
            if (!entry.is_directory()) continue;
            if (!recon_cfg().run_case.empty() && entry.path().filename() != recon_cfg().run_case) continue;
            if (!fs::is_regular_file(entry.path() / "cameras.txt") ||
                !fs::is_regular_file(entry.path() / "images.txt")) continue;
            cases.push_back(entry.path());
        }
        if (cases.empty()) throw std::runtime_error("No image pairs match the configuration");
        std::sort(cases.begin(), cases.end());
        pybind11::scoped_interpreter guard;
        std::string output_root_path = recon_cfg().output_root + "/";
        std::filesystem::create_directories(output_root_path);
        std::string runtime_txt = output_root_path + "runtimes.txt";
        std::ofstream fout(runtime_txt);
        if (!fout) throw std::runtime_error("Cannot write " + runtime_txt);
        fout << std::fixed << std::setprecision(1);
        fout << "case_name"
             << " det_ll_angle det_region_grow det_fit_extract det_total match_total"
             << " ours_curve_corr ours_curve_depth ours_curve_post ours_curve_total"
             << " ours_line_total ours_all_total"
             << " zhou_raw zhou_nurbs zhou_total"
             << " dlt_total\n";
        std::string data_root_path = recon_cfg().dataset_root;

        for (const auto& case_path : cases)
        {
            std::string case_folder = case_path.filename().string();

            std::cout<<"\nProcessing case ["<<case_folder<<"] ..."<<std::endl;

            std::string data_path = (fs::path(data_root_path) / case_folder).string() + "/";
            std::string camera_file = data_path + "cameras.txt";
            std::string list_file = data_path + "images.txt";
            std::vector<GroundTruth> gt_list;
            std::vector<ImageData> image_list;
            loadGroundTruthAndImageList(list_file, gt_list, image_list);
            std::cout << "Loaded images and ground truth data." << std::endl;

            CurveMatcher curve_matcher;
            CurveReconstructor reconstructor;

            for(int i = 0; i < image_list.size(); ++i)
            {
                std::string image_path = data_path + image_list[i].filename;
                if (!fs::is_regular_file(image_path)) throw std::runtime_error("Image not found: " + image_path);

                std::cout<<"Processing image [" <<image_list[i].filename<<"] ..."<<std::endl;

                const GroundTruth& gt = gt_list[i];
                int camera_id = image_list[i].camera_id;
                Camera camera;
                camera.load_intrinsics(camera_file, camera_id, recon_cfg().image_scale);

                curve_matcher.process(image_path, camera);
                cv::Mat denseMap = curve_matcher.get_dense_map();
                std::vector<CFeature> features = curve_matcher.get_features();

                if (i == 0) {
                    reconstructor.add_frame(0, features, gt.position, gt.quaternion, camera);
                } else if (i == 1) {
                    reconstructor.add_frame(1, features, gt.position, gt.quaternion, camera, denseMap);
                }
            }
            DetectionRuntimeStats det_stats;
            curve_matcher.get_case_detection_stats(det_stats);

            {
                reconstructor.apply_pose_error();
                reconstructor.process();
                reconstructor.process_Zhou();
                reconstructor.process_DLT();

                RuntimeStats stats;
                reconstructor.get_runtimes(stats);
                double matching_runtime = 0.0;
                curve_matcher.get_matching_runtime(matching_runtime);
                fout << case_folder
                     << " " << det_stats.ll_angle
                     << " " << det_stats.region_grow
                     << " " << det_stats.fit_extract
                     << " " << det_stats.total
                     << " " << matching_runtime
                     << " " << stats.ours_curve_corr
                     << " " << stats.ours_curve_depth
                     << " " << stats.ours_curve_post
                     << " " << stats.ours_curve_total
                     << " " << stats.ours_line_total
                     << " " << stats.ours_all_total
                     << " " << stats.zhou_raw
                     << " " << stats.zhou_nurbs
                     << " " << stats.zhou_total
                     << " " << stats.dlt_total
                     << "\n";

                std::cout << std::fixed << std::setprecision(1)
                          << "[Reconstruction runtime stats (ms)]: det_total=" << det_stats.total
                          << " match_total=" << matching_runtime
                          << " ours_all_total=" << stats.ours_all_total
                          << " zhou_total=" << stats.zhou_total
                          << " dlt_total=" << stats.dlt_total
                          << std::endl;

                std::string case_output_path = output_root_path + case_folder + "/";
                std::filesystem::create_directories(case_output_path);
                reconstructor.save_results(case_output_path, 0.005);
            }
        }
        fout.close();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << '\n';
        return 1;
    }
}
