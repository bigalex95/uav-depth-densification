#!/bin/bash
set -e

# ============================================
# Install system dependencies for the UAV depth densification
# pipeline.
#
# Required C++ deps (CMakeLists.txt):
#   - Eigen3          (linear algebra)
#   - GoogleTest      (unit tests)
#   - nlohmann-json   (JSON parsing)
#   - libtiff         (TIFF depth map I/O)
#
# Other tools:
#   - COLMAP          (Structure-from-Motion)
#   - cmake, g++      (build system)
#   - uv              (Python package manager)
# ============================================

show_help() {
    cat << 'EOF'
Usage: install_deps.sh [OPTIONS]

Install system-level dependencies for the depth densification pipeline.
Detects your package manager (apt/pacman/dnf) and installs all required
C++ libraries, build tools, and COLMAP.

Options:
    -h, --help      Show this help message and exit

Required C++ dependencies (from CMakeLists.txt):
    Eigen3           Linear algebra library
    GoogleTest       C++ testing framework
    nlohmann-json    JSON parser
    libtiff          TIFF image I/O (for depth maps)


Other tools:
    COLMAP           Structure-from-Motion pipeline
    cmake (>= 3.22)  Build system
    g++ (C++17)      Compiler
    uv               Python package manager
EOF
    exit 0
}

# Parse arguments
for arg in "$@"; do
    case $arg in
        -h|--help) show_help ;;
    esac
done

echo "Installing system dependencies..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    echo "Detected apt (Debian/Ubuntu)"
    sudo apt-get update
    sudo apt-get install -y \
        colmap \
        cmake \
        g++ \
        libeigen3-dev \
        nlohmann-json3-dev \
        libgtest-dev \
        libtiff-dev \
        wget \
        unzip

    # Build GoogleTest (required on some systems)
    if [ ! -f /usr/lib/libgtest.a ]; then
        echo "Building GoogleTest..."
        cd /usr/src/gtest
        sudo cmake .
        sudo make
        sudo cp lib/*.a /usr/lib/ 2>/dev/null || sudo cp *.a /usr/lib/ 2>/dev/null || true
    fi

elif command -v pacman &> /dev/null; then
    echo "Detected pacman (Arch Linux)"
    sudo pacman -S --noconfirm \
        colmap \
        cmake \
        gcc \
        eigen \
        nlohmann-json \
        gtest \
        libtiff \
        wget \
        unzip

elif command -v dnf &> /dev/null; then
    echo "Detected dnf (Fedora/RHEL)"
    sudo dnf install -y \
        colmap \
        cmake \
        gcc-c++ \
        eigen3-devel \
        json-devel \
        gtest-devel \
        libtiff-devel \
        wget \
        unzip

else
    echo "Warning: Unknown package manager. Please install manually:"
    echo "  Required:"
    echo "    - colmap            (Structure-from-Motion)"
    echo "    - cmake (>= 3.22)   (build system)"
    echo "    - g++ (C++17)       (compiler)"
    echo "    - Eigen3            (libeigen3-dev)"
    echo "    - nlohmann-json     (nlohmann-json3-dev)"
    echo "    - GoogleTest        (libgtest-dev)"
    echo "    - libtiff           (libtiff-dev)"
    echo "    - wget, unzip"
    echo ""
fi

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source ~/.bashrc"
fi

echo ""
echo "System dependencies installed successfully!"
echo "Next steps:"
echo "  1. uv sync                    # Install Python deps"
echo "  2. bash scripts/build.sh      # Build C++ components"
echo "  3. cd build && ctest          # Run C++ unit tests"
