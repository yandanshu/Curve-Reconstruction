#include "depth_rbf.hpp"
#include "recon_config.hpp"
#include "tic_toc.h"
#include <iostream>
#include <iomanip>

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return m;
}

double angle_difference(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {

    Eigen::Vector2d dir1(x1[0], x1[1]);
    Eigen::Vector2d dir2(x2[0], x2[1]);

    double cos_theta = dir1.dot(dir2) / (dir1.norm() * dir2.norm());

    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    double angle_rad = std::acos(cos_theta);
    double angle_deg = angle_rad * 180.0 / M_PI;

    if (angle_deg > 90.0) angle_deg = 180.0 - angle_deg;

    return angle_deg;
}

std::vector<double> KMeans1D(const std::vector<double>& data, int k, int max_iters = 100, double tol = 1e-6)
{
    int n = data.size();
    if(n == 0 || k <= 0) return {};

    std::vector<double> centers(k);
    for(int i = 0; i < k; ++i)
        centers[i] = data[i * (n-1) / (k-1)];

    std::vector<int> labels(n, 0);

    for(int iter = 0; iter < max_iters; ++iter)
    {
        bool changed = false;

        for(int i = 0; i < n; ++i)
        {
            double min_dist = std::numeric_limits<double>::max();
            int best = 0;
            for(int j = 0; j < k; ++j)
            {
                double dist = std::abs(data[i] - centers[j]);
                if(dist < min_dist) { min_dist = dist; best = j; }
            }
            if(labels[i] != best) { labels[i] = best; changed = true; }
        }

        std::vector<double> new_centers(k, 0.0);
        std::vector<int> counts(k, 0);
        for(int i = 0; i < n; ++i)
        {
            new_centers[labels[i]] += data[i];
            counts[labels[i]]++;
        }
        for(int j = 0; j < k; ++j)
            if(counts[j] > 0) new_centers[j] /= counts[j];

        double max_shift = 0.0;
        for(int j = 0; j < k; ++j)
            max_shift = std::max(max_shift, std::abs(new_centers[j] - centers[j]));

        centers = new_centers;
        if(max_shift < tol) break;
        if(!changed) break;
    }

    std::sort(centers.begin(), centers.end());

    return centers;
}

void approximate_depth_with_GaussianRBF(const std::vector<Eigen::Vector3d>& p1, const std::vector<Eigen::Vector3d>& p2,
                        const std::vector<double>& s1, const std::vector<double>& s2,
                        const std::vector<double>& w1, const std::vector<double>& w2,
                        const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                        const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                        const int num_RBF,
                        Eigen::VectorXd& coeffs1, Eigen::VectorXd& coeffs2,
                        std::vector<double>& centers1, std::vector<double>& centers2,
                        std::vector<double>& sigmas1, std::vector<double>& sigmas2,
                        double& residual)
{

    assert(s1.size() == s2.size());
    assert(p1.size() == p2.size() && p1.size() == s1.size());
    assert(w1.size() == w2.size() && w1.size() == s1.size());
    int n = p1.size();

    TicToc tt_km;
    centers1 = KMeans1D(s1, num_RBF);
    centers2 = KMeans1D(s2, num_RBF);

    sigmas1.resize(num_RBF, 0.0);
    sigmas2.resize(num_RBF, 0.0);
    const double kappa = recon_cfg().k_rbf;
    for(int i = 0; i < num_RBF; ++i)
    {
        if(i == 0)
        {
            sigmas1[i] = kappa * (centers1[i + 1] - centers1[i]);
            sigmas2[i] = kappa * (centers2[i + 1] - centers2[i]);
        }
        else if(i == num_RBF - 1)
        {
            sigmas1[i] = kappa * (centers1[i] - centers1[i - 1]);
            sigmas2[i] = kappa * (centers2[i] - centers2[i - 1]);
        }
        else
        {
            sigmas1[i] = kappa * (centers1[i + 1] - centers1[i - 1]) / 2;
            sigmas2[i] = kappa * (centers2[i + 1] - centers2[i - 1]) / 2;
        }
    }

    Eigen::MatrixXd A(3 * n, num_RBF * 2);
    Eigen::VectorXd b(3 * n);

    TicToc tt_build;
    int idx = 0;
    for(int i = 0; i < n; ++i)
    {
        Eigen::VectorXd z1_term(num_RBF);
        Eigen::VectorXd z2_term(num_RBF);

        for(int j = 0; j < num_RBF; ++j)
        {
            z1_term(j) = std::exp(-std::pow(s1[i] - centers1[j], 2) / (2 * sigmas1[j] * sigmas1[j]));
            z2_term(j) = std::exp(-std::pow(s2[i] - centers2[j], 2) / (2 * sigmas2[j] * sigmas2[j]));
        }

        Eigen::Vector3d _p1 = R1 * p1[i], _p2 = R2 * p2[i];

        double w = w2[i];

        A.row(idx) << _p1.x() * z1_term.transpose() * w, -_p2.x() * z2_term.transpose() * w;
        b(idx++) = (t2.x() - t1.x()) * w;
        A.row(idx) << _p1.y() * z1_term.transpose() * w, -_p2.y() * z2_term.transpose() * w;
        b(idx++) = (t2.y() - t1.y()) * w;
        A.row(idx) << _p1.z() * z1_term.transpose() * w, -_p2.z() * z2_term.transpose() * w;
        b(idx++) = (t2.z() - t1.z()) * w;
    }

    TicToc tt_solve;
    Eigen::VectorXd l = A.colPivHouseholderQr().solve(b);

    coeffs1 = l.head(num_RBF);
    coeffs2 = l.tail(num_RBF);

    residual = (A * l - b).norm();

}

double get_approximated_depth(const std::vector<double>& centers, const std::vector<double>& sigmas,
                              const Eigen::VectorXd& coeffs, const double s)
{
    assert(centers.size() == sigmas.size() && centers.size() == coeffs.size());
    double depth = 0;
    for(int i = 0; i < centers.size(); ++i)
    {
        double phi = std::exp(-std::pow(s - centers[i], 2) / (2 * sigmas[i] * sigmas[i]));
        depth += coeffs[i] * phi;
    }

    return depth;
}
