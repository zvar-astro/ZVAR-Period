# Installation Guide

## Without Docker

Here's a detailed guide for installing and setting up `ZVAR-Period` on a Linux system (without Docker, see the next section if you wish to use it).

### Prerequisites

- Linux distribution (Debian-based, Red Hat-based, or other)
- Administrator (sudo) privileges
- Internet connection for downloading packages
- Compatible NVIDIA GPU with appropriate drivers installed

### Step 1: Install System Dependencies

For Debian-based distributions (Ubuntu, Debian, etc.):
```bash
# Update package lists
sudo apt-get update

# Install essential build tools and dependencies
sudo apt-get install -y git build-essential cmake libboost-all-dev nano python3-dev libhdf5-serial-dev wget
```

For Red Hat-based distributions (CentOS, Fedora, RHEL, etc.):
```bash
# Install essential build tools and dependencies
sudo dnf install -y git cmake gcc-c++ boost-devel nano python3-devel hdf5-devel wget
```

### Step 2: Install CUDA Toolkit

#### Option 1: Package Manager Installation

For Debian-based systems:
```bash
# Install CUDA toolkit from package manager
sudo apt-get install -y nvidia-cuda-toolkit
```

For Red Hat-based systems:
```bash
# Enable EPEL repository
sudo dnf install -y epel-release

# Install CUDA toolkit
sudo dnf install -y cuda
```

After installing via package manager, locate your CUDA installation:
```bash
# Find CUDA installation
which nvcc
# Usually at /usr/bin/nvcc or /usr/local/cuda/bin/nvcc
```

#### Option 2: Manual Installation

First, check your NVIDIA driver version to determine which CUDA version to install:
```bash
nvidia-smi
```

Once you know your driver version, check the CUDA compatibility table on NVIDIA's website to ensure CUDA 12.6 is compatible with your driver. If not, choose a compatible version.

For Debian-based systems:
```bash
# Download CUDA repository package (adjust URL based on your distribution and compatible CUDA version)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-debian12-12-6-local_12.6.0-560.28.03-1_amd64.deb -O cuda.deb

# Install repository package
sudo dpkg -i cuda.deb

# Copy CUDA keyring
sudo cp /var/cuda-repo-debian12-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update package lists again
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-6

# Clean up
rm cuda.deb
```

For Red Hat-based systems:
```bash
# Download CUDA repository package (adjust URL based on your distribution and compatible CUDA version)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-rhel8-12-6-local-12.6.0-560.28.03-1.x86_64.rpm -O cuda.rpm

# Install repository package
sudo rpm -i cuda.rpm

# Install CUDA toolkit
sudo dnf install -y cuda-toolkit-12-6

# Clean up
rm cuda.rpm
```

### Step 3: Install a Python Virtual Environment Manager

For this guide, we will use `uv` as the virtual environment manager. You can also use any other manager like `venv`, `virtualenv`, or `conda`.

```bash
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 4: Clone Repository and Set Up Directory Structure

```bash
# first don't forget to cd to the directory where you want to install the application
# Then clone your project repository (replace with your actual repository URL)
git clone https://github.com/zvar-astro/ZVAR-Period.git

# Create packages directory if it doesn't exist
mkdir -p packages
cd packages

# Clone required repositories
git clone https://github.com/gpufit/Gpufit.git
git clone https://github.com/catchorg/Catch2.git
mkdir Gpufit-build

# Set Gpufit to a specific, earlier version
cd Gpufit
git checkout c2f520d
```

### Step 5: Set Up Environment Variables

FPW needs to know about your hardware limitations in order to avoid using more memory than available. Also, we need to tell it where to find different packages required at compile time. This is done by setting up environment variables. 

Add these lines to your `~/.bashrc` or `~/.bash_profile` (adjust CUDA path based on your installation method):

```bash
# For manual CUDA installation:
export PATH="$HOME/.local/bin/:/usr/local/cuda/bin:$PATH"

# For package manager CUDA installation, adjust based on your 'which nvcc' result
# export PATH="$HOME/.local/bin/:/usr/bin:$PATH"

# to not require you to edit all the different paths we'll have to define
# let's set the path to the application once
export ZVAR_PERIOD_PATH="path_to_ZVAR-Period_on_your_system" # EDIT THIS LINE

# To compile FPW and it's dependencies
# You may need to adjust these paths based on your CUDA installation, and where you've cloned the repositories
export GPUFIT_INCLUDE_DIR="${ZVAR_PERIOD_PATH}/packages/Gpufit/Gpufit"
export GPUFIT_LIB_DIR="${ZVAR_PERIOD_PATH}/packages/Gpufit-build/Gpufit"
export CATCH2_DIR="${ZVAR_PERIOD_PATH}/packages/Catch2$"

# To run FPW (you can also just specify those later at run time, rather then setting them here)
# You'll have to adjust these values based on your hardware, but here are some reasonable defaults:
export N_PROCESSES=1 # Number of processes to run in parallel
export N_GPUS=1 # Number of GPUs to use
export TOTAL_CPU_CAP=0.8 # Memory usage cap for all processes
export CHUNK_CPU_CAP=0.8 # Memory usage cap for a single chunk of lightcurves
export GPU_CAP=0.9 # GPU memory usage cap

# Last but not least, we specify where our input data is located, and where the output data should go
# and where the binaries we've built are found
export DATA_DIR="${ZVAR_PERIOD_PATH}/data" # EDIT THIS LINE
export OUTPUT_DIR="${ZVAR_PERIOD_PATH}/output" # EDIT THIS LINE
export BINARIES_LOCATION="${ZVAR_PERIOD_PATH}/build" # EDIT THIS LINE
```

Apply the changes:

```bash
source ~/.bashrc
```

### Step 6: Set Up Python Virtual Environment

```bash
# Make sure you are in the main application directory (ZVAR-Period)

# Create Python virtual environment
uv venv --python=python3.10

# Activate virtual environment
source .venv/bin/activate

# Install basic Python packages
uv pip install setuptools numpy
```

### Step 7: Patch Gpufit with Custom Models

We need custom models that aren't included in Gpufit by default. These models are used for fitting the light curves
to do some baseline removal. They are location in the `extensions/Gpufit/models` directory of this repository.

```bash
# Copy custom model files from your repo to Gpufit directory
cp extensions/Gpufit/models/cubic_1d.cuh packages/Gpufit/Gpufit/models/
cp extensions/Gpufit/models/polynomial_7degree_1d.cuh packages/Gpufit/Gpufit/models/
cp extensions/Gpufit/models/polynomial_15degree_1d.cuh packages/Gpufit/Gpufit/models/
cp extensions/Gpufit/models/quadratic_1d.cuh packages/Gpufit/Gpufit/models/
cp extensions/Gpufit/models/models.cuh packages/Gpufit/Gpufit/models/

# Copy constants header
cp extensions/Gpufit/constants.h packages/Gpufit/Gpufit/
```

### Step 8: Build Gpufit

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Navigate to Gpufit build directory
cd packages/Gpufit-build

# Configure with CMake
cmake ../Gpufit

# Build
make
```

### Step 9: Building FPW

Let's first go back to the main application directory and build the main application.
```bash
# Return to application directory
cd ../..
```

Then we will build the main application:

```bash
# Activate virtual environment if not already active
    source .venv/bin/activate

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ../src

# Build
make
```

### Step 10: Run the Application

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Run the main application
python -u scripts/runFiles.py
```

### Important Notes
- The version of CUDA installed should match your NVIDIA driver capabilities. Check compatibility at NVIDIA's CUDA documentation site
- CUDA installations via package managers may place files in different locations than manual installations
- If using the package manager version of CUDA, you may need to adjust CMake paths accordingly
- The repository URL in Step 5 needs to be replaced with your actual repository URL
- All necessary files are assumed to be part of your Git repository
- There is a built-in filtering of known terrestrial frequencies from the frequency grid used to run FPW. Alternatively, a custom list of frequencies + width can be loaded from a CSV file with the `FREQUENCIES_TO_FLAG_FILE` environment variable. These will not complement the existing filtering, but replace it.

### Troubleshooting
- If building Gpufit fails with CUDA errors, verify your CUDA installation with `nvcc --version`
- For package manager CUDA installations, you may need to specify the CUDA path explicitly during CMake configuration: `cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ../Gpufit`

## Output HDF5 File Format

After processing, results are written to an HDF5 file located at:

```
{OUTPUT_DIR}/{input_file_stem}_results.h5
```

The file uses a flat structure — all datasets are stored directly in the root group (`/`) with no sub-groups.

### Datasets

| Dataset | Group | Data Type | Shape | Description |
|---|---|---|---|---|
| `psids` | `/` | `uint64` | `[N_src]` | Pan-STARRS source identifier for each light curve |
| `ra` | `/` | `float32` | `[N_src]` | Right ascension of each source (degrees) |
| `dec` | `/` | `float32` | `[N_src]` | Declination of each source (degrees) |
| `valid` | `/` | `float32` | `[N_src]` | Fraction of exposures retained after outlier filtering (0–1) |
| `bestFreqs` | `/` | `float32` | `[N_src * N_samples * N_peaks]` | Frequencies (cycles/day) of the top periodogram peaks, stored as a flattened 1-D array |
| `significance` | `/` | `float32` | `[N_src * N_samples * N_peaks]` | Periodogram power at each peak frequency, stored as a flattened 1-D array |
| `means` | `/` | `float32` | `[N_src * N_samples]` | Mean periodogram power for each source and binning scheme, stored as a flattened 1-D array |
| `stds` | `/` | `float32` | `[N_src * N_samples]` | Standard deviation of the periodogram for each source and binning scheme, stored as a flattened 1-D array |

### Dimensions

| Symbol | Default value | Description |
|---|---|---|
| `N_src` | — | Total number of sources in the input file |
| `N_samples` | `3` | Number of phase-binning schemes (corresponding to 20, 10, and 5 bins) |
| `N_peaks` | `32` | Maximum number of peaks stored per source per binning scheme |

### Array indexing

The multi-dimensional datasets (`bestFreqs`, `significance`, `means`, `stds`) are stored as flattened 1-D arrays. The mapping between a logical index `(i, j, k)` and the flat index is:

| Dataset | Flat index formula | Description of `(i, j, k)` |
|---|---|---|
| `bestFreqs` | `i * N_samples * N_peaks + j * N_peaks + k` | `i` = source, `j` = binning scheme, `k` = peak rank |
| `significance` | `i * N_samples * N_peaks + j * N_peaks + k` | `i` = source, `j` = binning scheme, `k` = peak rank |
| `means` | `i * N_samples + j` | `i` = source, `j` = binning scheme |
| `stds` | `i * N_samples + j` | `i` = source, `j` = binning scheme |

Peak entries where no significant peak was detected are stored as `-1.0` for `bestFreqs` and IEEE 754 negative infinity (`-inf`) for `significance`.

### Reading the output with Python

```python
import h5py
import numpy as np

with h5py.File("results.h5", "r") as f:
    psids   = f["psids"][:]        # shape (N_src,)
    ra      = f["ra"][:]           # shape (N_src,)
    dec     = f["dec"][:]          # shape (N_src,)
    valid   = f["valid"][:]        # shape (N_src,)

    N_samples = 3
    N_peaks   = 32

    # Reshape multi-dimensional arrays for convenience
    best_freqs   = f["bestFreqs"][:].reshape(-1, N_samples, N_peaks)   # (N_src, N_samples, N_peaks)
    significance = f["significance"][:].reshape(-1, N_samples, N_peaks) # (N_src, N_samples, N_peaks)
    means        = f["means"][:].reshape(-1, N_samples)                 # (N_src, N_samples)
    stds         = f["stds"][:].reshape(-1, N_samples)                  # (N_src, N_samples)
```
