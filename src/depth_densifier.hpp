#pragma once

#include <Eigen/Dense>
#include <vector>

namespace depth_densifier
{

    class PolynomialTransformer
    {
    public:
        explicit PolynomialTransformer(int degree); // degree >= 1
        int degree() const;
        int num_features() const; // M = C(3+k,k)

        // Build one feature row: (d_tilde, u_norm, v_norm)
        Eigen::VectorXd transform(double d_tilde, double u_norm, double v_norm) const;

        // Build the full design matrix \Psi for N anchor points.
        // anchors: Nx3 matrix, columns = [d_tilde, u_norm, v_norm]
        Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &anchors) const;

    private:
        int degree_;
        int num_features_;
    };

    struct AnchorPoint
    {
        double d_tilde;  // monocular relative depth at this pixel
        double u_norm;   // normalised pixel coord ū ∈ [-1, 1]
        double v_norm;   // normalised pixel coord v̄ ∈ [-1, 1]
        double d_metric; // SfM metric depth (ground truth for fitting)
    };

    class DepthDensifier
    {
    public:
        explicit DepthDensifier(int degree = 1);

        // Fit beta from anchor points. N must be >= num_features().
        // Returns false if the system is rank-deficient (< full column rank).
        bool fit(const std::vector<AnchorPoint> &anchors);

        // Apply beta to produce a dense depth map.
        // relative_depth: HxW matrix of D_tilde values
        // Returns HxW matrix of metric depth estimates d_hat
        Eigen::MatrixXd densify(const Eigen::MatrixXd &relative_depth,
                                int image_width, int image_height,
                                double cx, double cy) const;

        const Eigen::VectorXd &coefficients() const; // beta
        bool is_fitted() const;

    private:
        PolynomialTransformer transformer_;
        Eigen::VectorXd coefficients_;
        bool is_fitted_;
    };

    struct ResidualStats
    {
        double mae;
        double rmse;
        double max_error;
        int n_samples;
    };

    class ResidualAnalyzer
    {
    public:
        // Compute residual statistics between predictions and ground-truth depths.
        // predictions and ground_truth must have the same length.
        static ResidualStats compute(const std::vector<double> &predictions,
                                     const std::vector<double> &ground_truth);

        // Returns true if the image passes quality thresholds.
        // RMSE < rmse_threshold AND MAE < mae_threshold
        static bool passes_quality_check(const ResidualStats &stats,
                                         double rmse_threshold,
                                         double mae_threshold);
    };

} // namespace depth_densifier
