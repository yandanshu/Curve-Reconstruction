#include "curve.hpp"

void Curve::compute_arc_length()
{
    arc_length = LNLib::NurbsCurve::ApproximateLength(nurbs, LNLib::IntegratorType::GaussLegendre);
}

void Curve::compute_curvature(int num_samples)
{
    curvature.clear();
    curvature.resize(num_samples);

    double s0 = 0.0, s1 = 1.0;
    double ds = (s1 - s0) / (num_samples - 1);

    for (int i = 0; i < num_samples; ++i) {
        double s = s0 + i * ds;

        std::vector<LNLib::XYZ> derivs =  LNLib::NurbsCurve::ComputeRationalCurveDerivatives(nurbs, 2, s);

        double x = derivs[0].X(),   y = derivs[0].Y(),
               dx = derivs[1].X(),  dy = derivs[1].Y(),
               ddx = derivs[2].X(), ddy = derivs[2].Y();

        double numerator = std::abs(dx * ddy - dy * ddx);
        double denominator = std::pow(dx * dx + dy * dy, 1.5);

        double kappa = 0.0;
        if (denominator > 1e-10) {
            kappa = numerator / denominator;
        }

        curvature[i] = kappa;
    }
}
