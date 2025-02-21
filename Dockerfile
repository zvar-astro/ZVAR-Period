FROM debian:bookworm-slim

WORKDIR /app

SHELL ["/bin/bash", "-c"]

ENV PATH="/root/.local/bin/:/usr/local/cuda/bin:$PATH"
ENV EIGEN3_INCLUDE_DIR="/app/packages/eigen"
ENV GPUFIT_INCLUDE_DIR="/app/packages/Gpufit/Gpufit"
ENV GPUFIT_LIB_DIR="/app/packages/Gpufit-build/Gpufit"

RUN apt-get update && apt-get install -y wget && \
    wget --no-verbose https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb -O cuda-repo.debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    wget --no-verbose https://sh.rustup.rs -O /rustup-init.sh && \
    wget --no-verbose https://astral.sh/uv/install.sh -O /uv-installer.sh && \
    dpkg -i cuda-repo.debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    rm cuda-repo.debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y curl wget git cmake build-essential libboost-all-dev python3-dev cuda-toolkit-12-8 cargo libhdf5-serial-dev && \
    sh /rustup-init.sh -y && rm /rustup-init.sh && \
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

# have a command that just keeps the container alive so we can exec into it
CMD ["tail", "-f", "/dev/null"]
