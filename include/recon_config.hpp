#ifndef RECON_CONFIG_HPP
#define RECON_CONFIG_HPP

#include <string>

struct ReconConfig {
    int p_nurbs = 3;
    int w1 = 100;
    double w2 = 20.0;
    double w3 = 1.0;
    double k_rbf = 6.0;
    double s0 = 1024.0;
    double match_dis_thre = 15.0;
    double trans_error_m = 0.0;
    double rot_error_deg = 0.0;
    int pose_error_seed = 42;
    std::string dataset_root;
    double image_scale = 0.2;
    std::string output_root;
    std::string run_case;
};

const ReconConfig& recon_cfg();
void load_recon_config(const std::string& path);
std::string default_recon_config_path();

#endif
