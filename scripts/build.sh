#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --help
for arg in "$@"; do
    case $arg in
        -h|--help)
            cat << 'EOF'
Usage: build.sh [OPTIONS]

Build the C++ depth densification components using CMake.
Produces: build/densify_main (CLI tool) and build/depth_densifier_test (unit tests).

Options:
    -h, --help      Show this help message and exit

Requirements (install with scripts/install_deps.sh):
    cmake (>= 3.22), g++ (C++17), Eigen3, GoogleTest, nlohmann-json, libtiff

Output:
    build/densify_main              Depth densification CLI
    build/depth_densifier_test      Unit tests (run with: cd build && ctest)
    build/compile_commands.json     For IDE integration

Examples:
    bash scripts/build.sh                               # Build in Release mode
    cd build && ctest --output-on-failure                # Run tests after build
    ./build/densify_main --help                          # See densify_main options
EOF
            exit 0
            ;;
    esac
done

echo "Building C++ components..."

cd "$PROJECT_ROOT"

# Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

echo "Build complete!"
echo ""
echo "To run tests: cd build && ctest --output-on-failure"
