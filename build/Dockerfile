FROM nvidia/cuda:11.0-base-ubuntu18.04 as base

ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Add new Nvidia repo keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        git-core \
        g++ \
        vim \
        zip \
        zlib1g-dev \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libcairo2-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

# For Nvidia
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-11-0 \
        libcublas-11-0 \
        cuda-nvrtc-11-0 \
        libcufft-11-0 \
        libcurand-11-0 \
        libcusolver-11-0 \
        libcusparse-11-0 \
        libcudnn8=8.0.5.39-1+cuda11.0

# Install TensorRT if not building for PowerPC
# NOTE: libnvinfer uses cuda11.1 versions
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends \
        libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/extras/CUPTI/lib64:/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda-11.0/lib64/stubs/libcuda.so /usr/local/cuda-11.0/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda-11.0/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt update -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y python3.8 python3-pip

RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m pip install --upgrade pip 

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

ENV TF_CPP_MIN_LOG_LEVEL=0

RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip && \
    unzip protoc-3.13.0-linux-x86_64.zip -d /app/protobuf/

ENV PATH "$PATH:/app/protobuf/bin"

RUN git clone https://github.com/tensorflow/models.git && \
    cd /app/models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python3 -m pip install .

# Override tensorflow version installed by objection detection api
RUN pip3 install tensorflow==2.4.0 tf-models-official==2.4.0 tensorflow_io pyparsing pycairo

# Install google cloud SDK
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcl && bash /tmp/gcl --install-dir=~/gcloud --disable-prompts
ENV PATH="$PATH:/root/gcloud/google-cloud-sdk/bin"
