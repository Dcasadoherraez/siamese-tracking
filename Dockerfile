# Build using
# sudo docker build -t siamese-tracking .

# **************** https://github.com/anibali/docker-pytorch **************** 

FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
# Pytorch version: https://anaconda.org/pytorch/pytorch/files
# Torchvision version: https://anaconda.org/pytorch/torchvision/files?version=0.11.2&type=&page=3
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.10.1=py3.8_cuda10.2_cudnn7.6.5_0" \
    "torchvision=0.11.2=py38_cu102" \
 && conda clean -ya

RUN sudo apt-get update

# Install OpenCV 
RUN sudo apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

# Install YOLORT Object Detector dependencies
RUN pip install -U yolort
RUN pip install --upgrade pip setuptools==59.5.0 wheel
RUN conda install pycocotools -c conda-forge    
# RUN pip install 'git+https://github.com/ppwwyyxx/cocoapi.git#subdirectory=PythonAPI' 

# Install GOT10k dataset utils
RUN pip install --upgrade got10k

# Set the default command to python3
CMD ["python3"]

RUN pip freeze

WORKDIR /app

ENV PYTHONPATH="/usr/src/app:${PYTHONPATH}"

RUN echo $PYTHONPATH
RUN echo "DONE!"
