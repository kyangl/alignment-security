FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y \
    git \
    unzip \
    zip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx 

RUN pip3 install --upgrade pip

# Install Python packages
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    albumentations \
    torch torchvision numpy torchmetrics pandas python-dotenv timm gdown transformers torchattacks

RUN pip3 install --no-cache-dir git+https://github.com/brain-score/vision.git
RUN pip3 install spikingjelly

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt