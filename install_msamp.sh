#!/bin/bash

# MS-AMP Installation Script
# This script installs Microsoft's Automatic Mixed Precision (MS-AMP) library
# with support for multi-GPU training using MSCCL

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect GPU architecture
detect_gpu_arch() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ $GPU_INFO == *"A100"* ]]; then
            echo "80"
        elif [[ $GPU_INFO == *"H100"* ]]; then
            echo "90"
        else
            print_warning "Unknown GPU architecture: $GPU_INFO"
            print_warning "Defaulting to compute capability 80 (A100)"
            echo "80"
        fi
    else
        print_warning "nvidia-smi not found. Defaulting to compute capability 80 (A100)"
        echo "80"
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "git is not installed. Please install git first."
        exit 1
    fi
    
    # Check if python3 is installed
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed. Please install python3 first."
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    # Check if CUDA is available
    if ! command -v nvcc &> /dev/null; then
        print_warning "CUDA compiler (nvcc) not found. Make sure CUDA is properly installed."
    fi
    
    print_success "Prerequisites check completed"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install build tools
    sudo apt-get install -y build-essential devscripts debhelper fakeroot
    
    print_success "System dependencies installed"
}

# Clone MS-AMP repository
clone_repository() {
    print_status "Cloning MS-AMP repository..."
    
    # Remove existing directory if it exists
    if [ -d "MS-AMP" ]; then
        print_warning "MS-AMP directory already exists. Removing..."
        rm -rf MS-AMP
    fi
    
    # Clone the repository
    git clone https://github.com/Azure/MS-AMP.git
    cd MS-AMP
    
    # Initialize and update submodules
    git submodule update --init --recursive
    
    print_success "Repository cloned successfully"
}

# Build MSCCL for multi-GPU support
build_msccl() {
    print_status "Building MSCCL for multi-GPU support..."
    print_warning "This may take 7-40 minutes depending on your hardware..."
    
    cd third_party/msccl
    
    # Detect GPU architecture
    COMPUTE_CAP=$(detect_gpu_arch)
    print_status "Detected compute capability: $COMPUTE_CAP"
    
    # Build MSCCL
    make -j src.build NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
    
    # Build and install packages
    make pkg.debian.build
    sudo dpkg -i build/pkg/deb/libnccl2_*.deb
    sudo dpkg -i build/pkg/deb/libnccl-dev_2*.deb
    
    cd - # Return to MS-AMP root directory
    
    print_success "MSCCL built and installed successfully"
}

# Install MS-AMP
install_msamp() {
    print_status "Installing MS-AMP..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install MS-AMP from source
    python3 -m pip install .
    
    # Run post-install setup
    make postinstall
    
    print_success "MS-AMP installed successfully"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Find NCCL library path
    NCCL_PATHS=(
        "/usr/lib/x86_64-linux-gnu/libnccl.so"
        "/usr/local/lib/libnccl.so"
        "/usr/lib/libnccl.so"
    )
    
    NCCL_LIBRARY=""
    for path in "${NCCL_PATHS[@]}"; do
        if [ -f "$path" ]; then
            NCCL_LIBRARY="$path"
            break
        fi
    done
    
    if [ -z "$NCCL_LIBRARY" ]; then
        print_error "NCCL library not found. Please check your NCCL installation."
        exit 1
    fi
    
    print_status "Found NCCL library at: $NCCL_LIBRARY"
    
    # Create environment setup script
    cat > setup_msamp_env.sh << EOF
#!/bin/bash
# MS-AMP Environment Setup
export NCCL_LIBRARY="$NCCL_LIBRARY"
export LD_PRELOAD="/usr/local/lib/libmsamp_dist.so:\${NCCL_LIBRARY}:\${LD_PRELOAD}"
echo "MS-AMP environment variables set successfully"
EOF
    
    chmod +x setup_msamp_env.sh
    
    print_success "Environment setup script created: setup_msamp_env.sh"
    print_status "Run 'source setup_msamp_env.sh' before using MS-AMP"
}

# Main installation function
main() {
    print_status "Starting MS-AMP installation..."
    
    # Parse command line arguments
    SKIP_MSCCL=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-msccl)
                SKIP_MSCCL=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--skip-msccl] [-h|--help]"
                echo "  --skip-msccl    Skip MSCCL installation (single GPU only)"
                echo "  -h, --help      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_prerequisites
    install_system_deps
    clone_repository
    
    if [ "$SKIP_MSCCL" = false ]; then
        build_msccl
    else
        print_warning "Skipping MSCCL installation. Multi-GPU training will not be available."
    fi
    
    install_msamp
    setup_environment
    
    print_success "MS-AMP installation completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Run 'source setup_msamp_env.sh' to set up environment variables"
    echo "2. Test your installation with a simple MS-AMP script"
    echo ""
    print_status "For multi-GPU training, make sure you built MSCCL (not skipped)"
}

# Run main function with all arguments
main "$@"