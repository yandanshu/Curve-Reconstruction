#include "recon_config.hpp"

#include <yaml-cpp/yaml.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifndef CURVE_RECONSTRUCTION_ROOT
#define CURVE_RECONSTRUCTION_ROOT "."
#endif

namespace {
ReconConfig g_recon_cfg;

template <typename T>
T require_scalar(const YAML::Node& node, const char* key)
{
    if (!node[key] || !node[key].IsScalar()) {
        throw std::runtime_error(std::string("missing scalar: ") + key);
    }
    return node[key].as<T>();
}

std::string resolve_path(const std::filesystem::path& base, const std::string& value)
{
    if (value.empty()) throw std::runtime_error("path must not be empty");
    return std::filesystem::absolute(base / value).lexically_normal().string();
}
}

const ReconConfig& recon_cfg()
{
    return g_recon_cfg;
}

std::string default_recon_config_path()
{
    return std::string(CURVE_RECONSTRUCTION_ROOT) + "/configs/recon.yaml";
}

void load_recon_config(const std::string& path)
{
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node recon = root["reconstruction"];
    const YAML::Node dataset = root["dataset"];
    const YAML::Node run = root["run"];
    ReconConfig cfg;
    cfg.p_nurbs = require_scalar<int>(recon, "p_nurbs");
    cfg.w1 = require_scalar<int>(recon, "w1");
    cfg.w2 = require_scalar<double>(recon, "w2");
    cfg.w3 = require_scalar<double>(recon, "w3");
    cfg.k_rbf = require_scalar<double>(recon, "k_rbf");
    cfg.s0 = require_scalar<double>(recon, "s0");
    cfg.match_dis_thre = require_scalar<double>(recon, "match_dis_thre");
    const auto base = std::filesystem::absolute(path).parent_path();
    cfg.dataset_root = resolve_path(base, require_scalar<std::string>(dataset, "root"));
    cfg.image_scale = require_scalar<double>(dataset, "image_scale");
    cfg.output_root = resolve_path(base, require_scalar<std::string>(run, "output_root"));
    cfg.run_case = require_scalar<std::string>(run, "case");
    if (root["pose_error"]) {
        const auto pe = root["pose_error"];
        if (pe["trans_error_m"]) cfg.trans_error_m = pe["trans_error_m"].as<double>();
        if (pe["rot_error_deg"]) cfg.rot_error_deg = pe["rot_error_deg"].as<double>();
        if (pe["seed"]) cfg.pose_error_seed = pe["seed"].as<int>();
    }
    if (cfg.p_nurbs < 1 || cfg.w1 <= 0) {
        throw std::runtime_error("p_nurbs and w1 must be positive");
    }
    for (double value : {cfg.w2, cfg.w3, cfg.k_rbf, cfg.s0, cfg.match_dis_thre, cfg.image_scale}) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("reconstruction parameters and image_scale must be finite and positive");
        }
    }
    for (double value : {cfg.trans_error_m, cfg.rot_error_deg}) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("pose_error values must be finite and non-negative");
        }
    }
    if (!std::filesystem::is_directory(cfg.dataset_root)) {
        throw std::runtime_error("Dataset directory not found: " + cfg.dataset_root);
    }
    g_recon_cfg = std::move(cfg);
    std::cerr << "[config] loaded " << path << '\n';
}
