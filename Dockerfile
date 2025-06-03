FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    git

# Install pytorch 2.1.2
RUN git clone --branch v2.1.2 https://github.com/pytorch/pytorch.git
RUN cd pytorch && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt
RUN pip install "numpy<2.0" && cd pytorch && TORCH_CUDA_ARCH_LIST="8.6;9.0" python3 setup.py install

# MSAMP
RUN git clone https://github.com/Azure/MS-AMP.git && cd MS-AMP && git submodule update --init --recursive

#set cuda home
ENV CUDA_HOME=/usr/local/cuda-12.2
# Install MSAMP dependencies
RUN cd MS-AMP/third_party/msccl && \
    make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" \
    && make pkg.debian.build && dpkg -i build/pkg/deb/libnccl2_*.deb && dpkg -i build/pkg/deb/libnccl-dev_2*.deb

RUN cd MS-AMP && pip3 install -e . && make postinstall
