FROM ghcr.io/azure/msamp:main-cuda12.2

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
# Install Python packages
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt

# install jupyterlab
RUN pip3 install jupyterlab

# set up jupyterlab
RUN mkdir -p /workspace/notebooks
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
