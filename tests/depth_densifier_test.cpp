#include <gtest/gtest.h>
#include "depth_densifier.hpp"
#include <cmath>
#include <vector>

using namespace depth_densifier;

TEST(PolynomialTransformer, PolyTf_Deg1_NumFeatures)
{
    PolynomialTransformer tf(1);
    EXPECT_EQ(tf.num_features(), 4);
}

TEST(PolynomialTransformer, PolyTf_Deg2_NumFeatures)
{
    PolynomialTransformer tf(2);
    EXPECT_EQ(tf.num_features(), 10);
}

TEST(PolynomialTransformer, PolyTf_Deg3_NumFeatures)
{
    PolynomialTransformer tf(3);
    EXPECT_EQ(tf.num_features(), 20);
}

TEST(PolynomialTransformer, PolyTf_Deg1_BiasPresent)
{
    PolynomialTransformer tf(1);
    Eigen::VectorXd feat = tf.transform(5.0, 0.5, -0.5);
    EXPECT_DOUBLE_EQ(feat(0), 1.0);
}

TEST(PolynomialTransformer, PolyTf_FitTransform_Shape)
{
    PolynomialTransformer tf(2);
    Eigen::MatrixXd anchors(5, 3);
    anchors.setZero();
    Eigen::MatrixXd features = tf.fit_transform(anchors);

    EXPECT_EQ(features.rows(), 5);
    EXPECT_EQ(features.cols(), 10);
}

TEST(DepthDensifier, Densifier_FitDensify_Linear)
{
    DepthDensifier densifier(1);

    // Perfect affine relation: d_metric = 2.0 * d_tilde + 0.5 * u_norm + 1.0
    std::vector<AnchorPoint> anchors = {
        {1.0, 0.0, 0.0, 3.0},
        {2.0, 1.0, 0.0, 5.5},
        {0.5, -1.0, 0.0, 1.5},
        {1.0, 0.0, 1.0, 3.0}};

    EXPECT_TRUE(densifier.fit(anchors));

    Eigen::MatrixXd relative_depth(2, 2);
    relative_depth << 1.0, 2.0,
        0.5, 1.0;

    Eigen::MatrixXd metric_depth = densifier.densify(relative_depth, 2, 2, 1.0, 1.0);

    // Evaluate error manually at pixel (0,0) mapped to center
    // u=0, cx=1 => u_norm = -1.0. v=0, cy=1 => v_norm = -1.0. d_tilde = 1.0
    // d_metric = 2.0*(1.0) + 0.5*(-1.0) + 1.0 = 2.5
    EXPECT_NEAR(metric_depth(0, 0), 2.5, 1e-9);
}

TEST(DepthDensifier, Densifier_Fit_FalseIfNLtM)
{
    DepthDensifier densifier(2);                               // needs 10 features
    std::vector<AnchorPoint> anchors(5, {1.0, 0.0, 0.0, 2.0}); // Only 5 anchors
    EXPECT_FALSE(densifier.fit(anchors));
}

TEST(DepthDensifier, Densifier_Fit_FalseIfRankDef)
{
    DepthDensifier densifier(1); // needs 4 features
    // Duplicate identical points make rank = 1 < 4
    std::vector<AnchorPoint> anchors(5, {1.0, 0.0, 0.0, 2.0});
    EXPECT_FALSE(densifier.fit(anchors));
}

TEST(DepthDensifier, Densifier_NotFitted_Initially)
{
    DepthDensifier densifier(1);
    EXPECT_FALSE(densifier.is_fitted());
}

TEST(ResidualAnalyzer, Residual_MAE_Perfect)
{
    std::vector<double> preds = {1.0, 2.0, 3.0};
    std::vector<double> gt = {1.0, 2.0, 3.0};
    ResidualStats stats = ResidualAnalyzer::compute(preds, gt);
    EXPECT_DOUBLE_EQ(stats.mae, 0.0);
}

TEST(ResidualAnalyzer, Residual_RMSE_Known)
{
    std::vector<double> preds = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> gt = {2.0, 2.0, 4.0, 4.0};
    ResidualStats stats = ResidualAnalyzer::compute(preds, gt);

    // errors: -1, 0, -1, 0 -> squared: 1, 0, 1, 0 -> mean = 0.5 -> rmse = sqrt(0.5)
    EXPECT_NEAR(stats.rmse, std::sqrt(0.5), 1e-6);
}

TEST(ResidualAnalyzer, Residual_QualCheck_Bounds)
{
    ResidualStats stats = {1.0, 1.0, 1.5, 10}; // mae=1.0, rmse=1.0

    // At-threshold passes
    EXPECT_TRUE(ResidualAnalyzer::passes_quality_check(stats, 1.0, 1.0));
    // Just-above fails (checking failure bounds)
    EXPECT_FALSE(ResidualAnalyzer::passes_quality_check(stats, 0.99, 1.0)); // rmse threshold too tight
    EXPECT_FALSE(ResidualAnalyzer::passes_quality_check(stats, 1.0, 0.99)); // mae threshold too tight
}
