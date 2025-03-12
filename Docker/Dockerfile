# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Use bash for all RUN commands so that conda activation works properly
SHELL ["/bin/bash", "-c"]

# Set noninteractive mode for apt-get to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0.0

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Clone the vggsfm repository
RUN git clone https://github.com/Vlhermitte/acezero.git /opt/acezero

# Set working directory
WORKDIR /opt/acezero

# Run the installation script from the repository
ENV ENV_NAME=ace0

# Create the conda environment
RUN conda env create -f environment.yml

RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME && \
    python -m pip install -e dsacstar

# Default command: activate the conda environment and launch bash
CMD ["bash", "-c", "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate ace0 && exec bash"]