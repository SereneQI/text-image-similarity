FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive

ENV http_proxy=http://10.100.9.1:2001 https_proxy=http://10.100.9.1:2001
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    cython \
    bzip2 \
    libx11-6 \
    build-essential \
    wget \
    libgtk2.0-0 \
    libcanberra-gtk-module
    

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
   /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install Miniconda
#RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
#    chmod +x ~/miniconda.sh && \
#    ~/miniconda.sh -b -p ~/miniconda && \
#    rm ~/miniconda.sh
#ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /opt/conda/bin/conda install conda-build \
 && /opt/conda/bin/conda create -y --name py36 python=3.6.5

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:/opt/conda/bin/:$PATH

# CUDA 9.2-specific steps
RUN conda install -c pytorch pytorch torchvision gensim faiss-gpu theano nltk

# Install conda packages 
RUN conda install -y h5py=2.8.0 requests=2.19.1 graphviz=2.38.0
RUN conda install -y -c menpo opencv3=3.1.0
RUN conda install -y -c qwant fasttext-python

# Install pip packages
RUN pip install h5py-cache==1.0 cupy torchnet graphviz==0.8.4 tensorboardx


#install SRU for RNN
RUN pip install sru[cuda]

#pycocotools for MS COCO
RUN conda install -c hcc pycocotools

# Set the default command to python3
CMD ["python3"]
