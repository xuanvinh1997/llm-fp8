FROM ghcr.io/azure/msamp:main-cuda12.2

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt

# Install jupyterlab
RUN pip3 install jupyterlab

# Set up SSH for Vast.ai
RUN mkdir /var/run/sshd
RUN echo 'root:vastai' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Set up jupyterlab
RUN mkdir -p /workspace/notebooks

# Create startup script
RUN echo '#!/bin/bash\n\
# Start SSH daemon\n\
service ssh start\n\
\n\
# Start Jupyter Lab\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --ServerApp.token="" --ServerApp.password="" &\n\
\n\
# Keep container running\n\
tail -f /dev/null' > /start.sh && chmod +x /start.sh

# Expose ports
EXPOSE 22 8888

# Use the startup script
CMD ["/start.sh"]