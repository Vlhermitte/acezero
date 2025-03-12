FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y wget \
    git libnvidia-egl-wayland1 \
    xvfb mesa-utils libgl1-mesa-glx \
    libgl1-mesa-dri libglib2.0-0

# Install Miniforge
RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

RUN git clone https://github.com/nianticlabs/acezero --recursive

RUN conda install ipykernel

# Create ace0 env
RUN cd acezero && \
    conda env create -f environment.yml

RUN echo "source activate ace0" > ~/.bashrc && \
    conda run -n ace0 pip install ipykernel && \
    conda install -n ace0 -c conda-forge libstdcxx-ng && \
    /opt/conda/envs/ace0/bin/python -m ipykernel install --user --name=ace0 \

RUN conda run -n ace0 python -m pip install joblib scipy scikit-learn

RUN cd acezero/dsacstar && \
    conda run -n ace0 python setup.py install

# Set up Xvfb
ENV DISPLAY=:99
RUN Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99

WORKDIR /workspace/acezero

CMD ["bash", "-c", "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate ace0 && exec bash"]