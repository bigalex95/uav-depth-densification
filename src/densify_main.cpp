/**
 * @file densify_main.cpp
 * @brief CLI for depth densification using the C++ DepthDensifier.
 *
 * Reads SfM reconstruction JSON and monocular depth TIFFs, fits a polynomial
 * model to SfM anchor points, and produces densified depth maps.
 *
 * Usage:
 *   ./densify_main [--recon PATH] [--depth-dir PATH] [--output-dir PATH] [--degree N]
 */

#include "depth_densifier.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tiffio.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace depth_densifier;

// =============================================================================
// TIFF I/O
// =============================================================================

/**
 * @brief Read a 32-bit float TIFF depth map.
 * @param path Path to .tif file
 * @param width Output image width
 * @param height Output image height
 * @return Vector of float depth values (row-major)
 */
std::vector<float> read_tiff(const fs::path &path, int &width, int &height)
{
    TIFF *tif = TIFFOpen(path.c_str(), "r");
    if (!tif)
        throw std::runtime_error("Cannot open TIFF: " + path.string());

    uint32_t w, h;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    width = static_cast<int>(w);
    height = static_cast<int>(h);

    std::vector<float> data(w * h);
    for (uint32_t row = 0; row < h; ++row)
    {
        TIFFReadScanline(tif, data.data() + row * w, row);
    }
    TIFFClose(tif);
    return data;
}

/**
 * @brief Write a 32-bit float TIFF depth map.
 */
void write_tiff(const fs::path &path, const std::vector<float> &data, int width, int height)
{
    TIFF *tif = TIFFOpen(path.c_str(), "w");
    if (!tif)
        throw std::runtime_error("Cannot create TIFF: " + path.string());

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(height));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);

    for (int row = 0; row < height; ++row)
    {
        TIFFWriteScanline(tif, const_cast<float *>(data.data() + row * width), row);
    }
    TIFFClose(tif);
}

// =============================================================================
// 3D-to-2D projection (mirrors the Python version)
// =============================================================================

struct Projection
{
    std::vector<Eigen::Vector2d> uv;
    std::vector<double> depths;
};

/**
 * @brief Project 3D points onto an image and return visible (u,v) + depth.
 */
Projection project_points(
    const Eigen::MatrixXd &pts, // (N,3)
    const Eigen::Matrix3d &R,
    const Eigen::Vector3d &t,
    const Eigen::Matrix3d &K,
    int width, int height)
{
    Projection proj;
    for (int i = 0; i < pts.rows(); ++i)
    {
        Eigen::Vector3d X_cam = R * pts.row(i).transpose() + t;
        if (X_cam(2) <= 0.1)
            continue;

        Eigen::Vector3d x = K * X_cam;
        double u = x(0) / x(2);
        double v = x(1) / x(2);

        if (u >= 0 && u < width && v >= 0 && v < height)
        {
            proj.uv.emplace_back(u, v);
            proj.depths.push_back(X_cam(2));
        }
    }
    return proj;
}

// =============================================================================
// CLI argument parsing
// =============================================================================

struct Args
{
    fs::path recon = "sfm_output/reconstruction.json";
    fs::path depth_dir = "depth_maps";
    fs::path output_dir = "densified_maps";
    int degree = 2;
};

Args parse_args(int argc, char *argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "--recon") && i + 1 < argc)
            a.recon = argv[++i];
        else if ((arg == "--depth-dir") && i + 1 < argc)
            a.depth_dir = argv[++i];
        else if ((arg == "--output-dir") && i + 1 < argc)
            a.output_dir = argv[++i];
        else if ((arg == "--degree") && i + 1 < argc)
            a.degree = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "densify_main - C++ depth densification using polynomial OLS regression\n"
                << "\n"
                << "Reads an SfM reconstruction JSON and monocular depth TIFFs, projects sparse\n"
                << "3D points as anchors, fits a polynomial model to align relative depths to\n"
                << "metric scale, and produces densified depth maps for every registered image.\n"
                << "\n"
                << "Usage: densify_main [OPTIONS]\n"
                << "\n"
                << "Options:\n"
                << "  -h, --help           Show this help message and exit\n"
                << "  --recon PATH         Path to reconstruction JSON file\n"
                << "                       (default: sfm_output/reconstruction.json)\n"
                << "  --depth-dir PATH     Directory containing monocular depth TIFFs from\n"
                << "                       run_depth.py. Each TIFF should be 32-bit float.\n"
                << "                       (default: depth_maps)\n"
                << "  --output-dir PATH    Output directory for densified depth TIFFs.\n"
                << "                       Subdirectories are created to match depth-dir structure.\n"
                << "                       Also writes densification_summary.json with per-image\n"
                << "                       MAE, RMSE, and anchor count statistics.\n"
                << "                       (default: densified_maps)\n"
                << "  --degree N           Polynomial degree for the OLS regression (1-3).\n"
                << "                       Degree 1 (4 features):  global scale + shift\n"
                << "                       Degree 2 (10 features): + spatial corrections [recommended]\n"
                << "                       Degree 3 (20 features): + higher-order corrections\n"
                << "                       (default: 2)\n"
                << "\n"
                << "Examples:\n"
                << "  densify_main                                 # Use all defaults\n"
                << "  densify_main --output-dir densified_cpp      # Custom output directory\n"
                << "  densify_main --degree 1                      # Linear (scale+shift) only\n"
                << "  densify_main --depth-dir depth_maps --degree 3\n"
                << "\n"
                << "Output:\n"
                << "  <output-dir>/<subfolder>/<image_stem>.tif     Densified depth TIFFs\n"
                << "  <output-dir>/densification_summary.json       Per-image metrics (MAE, RMSE)\n";
            std::exit(0);
        }
    }
    return a;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char *argv[])
{
    Args args = parse_args(argc, argv);

    // Load reconstruction JSON
    std::ifstream ifs(args.recon);
    if (!ifs.is_open())
    {
        std::cerr << "Error: cannot open " << args.recon << "\n";
        return 1;
    }
    json recon = json::parse(ifs);
    ifs.close();

    std::cout << "Loaded reconstruction: "
              << recon["num_images"] << " images, "
              << recon["num_points"] << " 3D points\n";

    // Parse 3D points into Eigen matrix
    auto &pts_json = recon["points3D"];
    int num_pts = static_cast<int>(pts_json.size());
    Eigen::MatrixXd pts_3d(num_pts, 3);
    for (int i = 0; i < num_pts; ++i)
    {
        auto &xyz = pts_json[i]["xyz"];
        pts_3d(i, 0) = xyz[0].get<double>();
        pts_3d(i, 1) = xyz[1].get<double>();
        pts_3d(i, 2) = xyz[2].get<double>();
    }

    // Camera dimensions
    std::map<int, std::pair<int, int>> cam_dims;
    for (auto &[cid, cam] : recon["cameras"].items())
    {
        cam_dims[std::stoi(cid)] = {cam["width"].get<int>(), cam["height"].get<int>()};
    }

    fs::create_directories(args.output_dir);

    json summary_results = json::array();
    int processed = 0, failed = 0;

    for (auto &[img_id, img_data] : recon["images"].items())
    {
        std::string name = img_data["name"].get<std::string>();
        std::cout << "\nProcessing " << name << "...\n";

        // Parse R, t, K
        Eigen::Matrix3d R, K;
        Eigen::Vector3d t;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                R(r, c) = img_data["R"][r][c].get<double>();
                K(r, c) = img_data["K"][r][c].get<double>();
            }
        t << img_data["t"][0].get<double>(),
            img_data["t"][1].get<double>(),
            img_data["t"][2].get<double>();

        int cam_id = img_data["camera_id"].get<int>();
        auto [width, height] = cam_dims[cam_id];
        double cx = K(0, 2);
        double cy = K(1, 2);

        // Project 3D points
        auto proj = project_points(pts_3d, R, t, K, width, height);
        std::cout << "  Projected " << proj.uv.size() << " anchor points\n";

        // Load monocular depth TIFF
        std::string depth_name = name;
        // Replace any image extension with .tif
        auto pos = depth_name.rfind('.');
        if (pos != std::string::npos)
            depth_name = depth_name.substr(0, pos) + ".tif";

        fs::path depth_path = args.depth_dir / depth_name;
        if (!fs::exists(depth_path))
        {
            std::cout << "  SKIP: depth map not found at " << depth_path << "\n";
            continue;
        }

        int dw, dh;
        auto mono_depth = read_tiff(depth_path, dw, dh);
        std::cout << "  Loaded depth map: " << dw << "x" << dh << "\n";

        // Build anchors with new interface:
        // d_tilde = monocular depth (relative)
        // u_norm = (u - cx) / (width/2)
        // v_norm = (v - cy) / (height/2)
        // d_metric = metric depth from SfM
        double half_w = dw / 2.0;
        double half_h = dh / 2.0;

        std::vector<AnchorPoint> anchors;
        std::vector<double> gt_depths; // Ground truth depths for residual computation

        for (size_t i = 0; i < proj.uv.size(); ++i)
        {
            int ui = std::clamp(static_cast<int>(proj.uv[i](0)), 0, dw - 1);
            int vi = std::clamp(static_cast<int>(proj.uv[i](1)), 0, dh - 1);
            float mono_val = mono_depth[vi * dw + ui];
            if (mono_val > 1e-6)
            {
                double u_norm = (proj.uv[i](0) - cx) / half_w;
                double v_norm = (proj.uv[i](1) - cy) / half_h;
                double d_tilde = static_cast<double>(mono_val);
                double d_metric = proj.depths[i];
                anchors.push_back({d_tilde, u_norm, v_norm, d_metric});
                gt_depths.push_back(d_metric);
            }
        }
        std::cout << "  Valid anchors (with mono depth): " << anchors.size() << "\n";

        if (anchors.size() < 4)
        {
            std::cout << "  FAILED: insufficient anchors (need at least 4 for degree 1)\n";
            ++failed;
            continue;
        }

        // Fit densifier
        DepthDensifier densifier(args.degree);
        if (!densifier.fit(anchors))
        {
            std::cout << "  FAILED: polynomial fitting failed\n";
            ++failed;
            continue;
        }

        // Build relative depth matrix and densify
        Eigen::MatrixXd relative_depth(dh, dw);
        for (int vv = 0; vv < dh; ++vv)
        {
            for (int uu = 0; uu < dw; ++uu)
            {
                relative_depth(vv, uu) = static_cast<double>(mono_depth[vv * dw + uu]);
            }
        }

        Eigen::MatrixXd dense_depth = densifier.densify(relative_depth, dw, dh, cx, cy);

        // Convert to float vector for output
        std::vector<float> densified(dw * dh);
        for (int vv = 0; vv < dh; ++vv)
        {
            for (int uu = 0; uu < dw; ++uu)
            {
                densified[vv * dw + uu] = static_cast<float>(dense_depth(vv, uu));
            }
        }

        // Compute residuals at anchor locations
        std::vector<double> predictions;
        predictions.reserve(anchors.size());
        for (size_t i = 0; i < anchors.size(); ++i)
        {
            // Find pixel location from normalized coords
            double u_px = anchors[i].u_norm * half_w + cx;
            double v_px = anchors[i].v_norm * half_h + cy;
            int ui = std::clamp(static_cast<int>(u_px), 0, dw - 1);
            int vi = std::clamp(static_cast<int>(v_px), 0, dh - 1);
            predictions.push_back(static_cast<double>(densified[vi * dw + ui]));
        }

        ResidualStats stats = ResidualAnalyzer::compute(predictions, gt_depths);
        std::cout << "  Densified: MAE=" << stats.mae
                  << ", RMSE=" << stats.rmse
                  << ", max_error=" << stats.max_error << "\n";

        // Save output TIFF
        fs::path output_path = args.output_dir / depth_name;
        fs::create_directories(output_path.parent_path());
        write_tiff(output_path, densified, dw, dh);
        std::cout << "  Saved to " << output_path << "\n";

        // Add to summary
        json entry;
        entry["image"] = name;
        entry["num_anchors"] = anchors.size();
        entry["success"] = true;
        entry["mae"] = stats.mae;
        entry["rmse"] = stats.rmse;
        entry["max_error"] = stats.max_error;
        summary_results.push_back(entry);
        ++processed;
    }

    // Compute aggregate statistics
    double avg_mae = 0, avg_rmse = 0;
    if (!summary_results.empty())
    {
        for (auto &r : summary_results)
        {
            avg_mae += r["mae"].get<double>();
            avg_rmse += r["rmse"].get<double>();
        }
        avg_mae /= static_cast<double>(summary_results.size());
        avg_rmse /= static_cast<double>(summary_results.size());
    }

    // Summary
    std::cout << "\n============================================================\n"
              << "SUMMARY\n"
              << "============================================================\n"
              << "Successfully processed: " << processed << " images\n"
              << "Failed: " << failed << " images\n";

    if (!summary_results.empty())
    {
        std::cout << "Average MAE:  " << avg_mae << "\n"
                  << "Average RMSE: " << avg_rmse << "\n";
    }

    // Build full summary JSON with aggregate stats
    json full_summary;
    full_summary["processed"] = processed;
    full_summary["failed"] = failed;
    full_summary["average_mae"] = avg_mae;
    full_summary["average_rmse"] = avg_rmse;
    full_summary["degree"] = args.degree;
    full_summary["images"] = summary_results;

    // Save summary JSON
    fs::path summary_path = args.output_dir / "densification_summary.json";
    std::ofstream ofs(summary_path);
    ofs << full_summary.dump(2);
    ofs.close();
    std::cout << "\nSummary saved to " << summary_path << "\n";

    // Save human-readable text summary
    fs::path txt_path = args.output_dir / "summary.txt";
    std::ofstream txt(txt_path);
    txt << "Depth Densification Summary\n";
    txt << "===========================\n\n";
    txt << "Processed: " << processed << " images\n";
    txt << "Failed:    " << failed << " images\n";
    txt << "Degree:    " << args.degree << "\n\n";
    txt << "Average MAE:  " << avg_mae << " m\n";
    txt << "Average RMSE: " << avg_rmse << " m\n";
    txt.close();
    std::cout << "Text summary saved to " << txt_path << "\n";

    return 0;
}
