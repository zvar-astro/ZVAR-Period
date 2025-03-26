FROM debian:bookworm-slim

WORKDIR /app

SHELL ["/bin/bash", "-c"]

ENV PATH="/root/.local/bin/:/usr/local/cuda/bin:$PATH"
ENV EIGEN3_INCLUDE_DIR="/app/packages/eigen"
ENV GPUFIT_INCLUDE_DIR="/app/packages/Gpufit/Gpufit"
ENV GPUFIT_LIB_DIR="/app/packages/Gpufit-build/Gpufit"

ENV CUDA_DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-debian12-12-6-local_12.6.0-560.28.03-1_amd64.deb"
ENV CUDA_APT_PACKAGE="cuda-toolkit-12-6"

RUN apt-get update && apt-get install -y wget && \
    wget --no-verbose https://astral.sh/uv/install.sh -O /uv-installer.sh && \
    wget --no-verbose ${CUDA_DEB_URL} -O cuda.deb && \
    dpkg -i cuda.deb && rm cuda.deb && \
    cp /var/cuda-repo-debian12-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y git build-essential cmake libboost-all-dev \
    nano python3-dev libhdf5-serial-dev ${CUDA_APT_PACKAGE} && \
    sh /uv-installer.sh && rm /uv-installer.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN uv venv --python=python3.10 && \
    source .venv/bin/activate && \
    uv pip install setuptools numpy && \
    mkdir -p packages && cd packages && \
    git clone https://gitlab.com/libeigen/eigen.git && \
    git clone https://github.com/gpufit/Gpufit.git && \
    mkdir Gpufit-build

COPY ["extensions/Gpufit/models/cubic_1d.cuh", \
    "extensions/Gpufit/models/polynomial_7degree_1d.cuh", \
    "extensions/Gpufit/models/polynomial_15degree_1d.cuh", \
    "extensions/Gpufit/models/quadratic_1d.cuh", \
    "extensions/Gpufit/models/models.cuh", \
    "packages/Gpufit/Gpufit/models/"]

COPY extensions/Gpufit/constants.h packages/Gpufit/Gpufit/

RUN source .venv/bin/activate && \
    cd packages/Gpufit-build && \
    cmake ../Gpufit && \
    make

COPY src src

RUN source .venv/bin/activate && \
    mkdir build && \
    cd build && \
    cmake ../src && \
    make

# # RUN FPW on all the files
CMD ["bash", "-c", "source .venv/bin/activate && cd src && python -u runFiles.py"]

# Keep the container running
# CMD ["bash", "-c", "tail -f /dev/null"]