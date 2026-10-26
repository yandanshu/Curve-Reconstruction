#ifndef POINTCLOUD_TO_NURBS_HPP
#define POINTCLOUD_TO_NURBS_HPP

#include "NurbsCurve.h"

#include <Eigen/Dense>
#include <vector>
#include <algorithm>

std::vector<Eigen::Vector3d> SortPointCloudByPrincipalDirection(const std::vector<Eigen::Vector3d>& points);

bool fit_points_to_nurbs(const std::vector<Eigen::Vector3d>& points,
                         LNLib::LN_NurbsCurve& nurbs,
                         int& ctrl_used,
                         int img_height,
                         int ctrl_override = -1);

#endif
