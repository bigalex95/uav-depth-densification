#include "depth_densifier.hpp"
#include <stdexcept>
#include <cmath>

namespace depth_densifier {

// ============================================================================
// PolynomialTransformer
// ============================================================================

PolynomialTransformer::PolynomialTransformer(int degree) : degree_(degree) {
    if (degree < 1) {
        throw std::invalid_argument("Degree must be >= 1");
    }
    // Number of terms in 3-variable polynomial of degree k: C(3+k, k) = (k+3)(k+2)(k+1)/6
    // Degree 1: 4 features   (1, d, u, v)
    // Degree 2: 10 features  (+ d², du, dv, u², uv, v²)
    // Degree 3: 20 features  (+ cubic terms)
    num_features_ = (degree + 3) * (degree + 2) * (degree + 1) / 6;
}

int PolynomialTransformer::degree() const { return degree_; }
int PolynomialTransformer::num_features() const { return num_features_; }

Eigen::VectorXd PolynomialTransformer::transform(double d_tilde, double u_norm, double v_norm) const {
    Eigen::VectorXd feat(num_features_);
    int idx = 0;
    // Graded lexicographic ordering
    for (int td = 0; td <= degree_; ++td) {
        for (int p = td; p >= 0; --p) {
            for (int q = td - p; q >= 0; --q) {
                int r = td - p - q;
                feat(idx++) = std::pow(d_tilde, p) * std::pow(u_norm, q) * std::pow(v_norm, r);
            }
        }
    }
    return feat;
}

Eigen::MatrixXd PolynomialTransformer::fit_transform(const Eigen::MatrixXd& anchors) const {
    const int N = static_cast<int>(anchors.rows());
    Eigen::MatrixXd Phi(N, num_features_);
    for (int i = 0; i < N; ++i) {
        Phi.row(i) = transform(anchors(i, 0), anchors(i, 1), anchors(i, 2));
    }
    return Phi;
}

// ============================================================================
// DepthDensifier
// ============================================================================

DepthDensifier::DepthDensifier(int degree) : transformer_(degree), is_fitted_(false) {}

bool DepthDensifier::fit(const std::vector<AnchorPoint>& anchors) {
    const int N = static_cast<int>(anchors.size());
    const int M = transformer_.num_features();

    if (N < M) {
        is_fitted_ = false;
        return false;
    }

    Eigen::MatrixXd X(N, 3);
    Eigen::VectorXd y(N);

    for (int i = 0; i < N; ++i) {
        X(i, 0) = anchors[i].d_tilde;
        X(i, 1) = anchors[i].u_norm;
        X(i, 2) = anchors[i].v_norm;
        y(i) = anchors[i].d_metric;
    }

    Eigen::MatrixXd Phi = transformer_.fit_transform(X);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Phi);

    if (qr.rank() < M) {
        is_fitted_ = false;
        return false;
    }

    coefficients_ = qr.solve(y);
    is_fitted_ = true;
    return true;
}

Eigen::MatrixXd DepthDensifier::densify(const Eigen::MatrixXd& relative_depth,
                                        int image_width, int image_height,
                                        double cx, double cy) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model not fitted. Call fit() first.");
    }

    Eigen::MatrixXd dense_depth(image_height, image_width);
    const double half_w = image_width / 2.0;
    const double half_h = image_height / 2.0;

    for (int v = 0; v < image_height; ++v) {
        double v_norm = (v - cy) / half_h;
        for (int u = 0; u < image_width; ++u) {
            double u_norm = (u - cx) / half_w;
            double d_tilde = relative_depth(v, u);

            Eigen::VectorXd feat = transformer_.transform(d_tilde, u_norm, v_norm);
            dense_depth(v, u) = feat.dot(coefficients_);
        }
    }
    return dense_depth;
}

const Eigen::VectorXd& DepthDensifier::coefficients() const { return coefficients_; }
bool DepthDensifier::is_fitted() const { return is_fitted_; }

// ============================================================================
// ResidualAnalyzer
// ============================================================================

ResidualStats ResidualAnalyzer::compute(const std::vector<double>& predictions,
                                        const std::vector<double>& ground_truth) {
    if (predictions.size() != ground_truth.size()) {
        throw std::invalid_argument("Size mismatch between predictions and ground truth.");
    }

    const int N = static_cast<int>(predictions.size());
    if (N == 0) return {0.0, 0.0, 0.0, 0};

    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double max_err = 0.0;

    for (int i = 0; i < N; ++i) {
        double err = std::abs(predictions[i] - ground_truth[i]);
        sum_abs_err += err;
        sum_sq_err += err * err;
        if (err > max_err) {
            max_err = err;
        }
    }

    ResidualStats stats;
    stats.mae = sum_abs_err / N;
    stats.rmse = std::sqrt(sum_sq_err / N);
    stats.max_error = max_err;
    stats.n_samples = N;
    return stats;
}

bool ResidualAnalyzer::passes_quality_check(const ResidualStats& stats,
                                            double rmse_threshold,
                                            double mae_threshold) {
    // Uses <= to satisfy "at-threshold passes" test requirement
    return stats.rmse <= rmse_threshold && stats.mae <= mae_threshold;
}

} // namespace depth_densifier
