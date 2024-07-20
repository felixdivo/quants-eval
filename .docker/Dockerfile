# syntax = docker/dockerfile:1.2
ARG KIND=3.11
# FROM lmsysorg/sglang:latest 
FROM mkrausio/ml_research:base-3.11-cuda-12.1.0
# FROM mkrausio/ml_research:torch-v2.3.1-3.11-cuda-12.1.0


ARG COMPOSE_PROJ_NAME="ts-qa"
ARG GITHUB_TOKEN
ARG CUDA_VERSION=12.1.0

ENV GITHUB_TOKEN=${GITHUB_TOKEN}

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt autoremove python3 -y &&\
  rm -rf /usr/bin/python3 /usr/bin/pip3 /bin/python3 /bin/pip3 &&\
  ln -s /usr/bin/python${KIND} /usr/bin/python3 &&\
  ln -s /usr/bin/pip${KIND} /usr/bin/pip3 

RUN apt update && apt install git-lfs && git lfs install

RUN mkdir -p workspaces/$PROJECT_NAME

WORKDIR /workspaces/$PROJECT_NAME

COPY requirements.txt .

RUN ldconfig /usr/local/cuda-12.1/compat/

RUN --mount=type=cache,target=/root/.cache/pip \
python -m pip install --upgrade pip && \
pip install "torch==2.3.0" -r requirements.txt 

RUN pip install "sglang[all]"
# # Install FlashInfer CUDA kernels
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

RUN pip install -U setuptools


# RUN mkdir -p /data && cd /data && wget https://nlp.stanford.edu/data/glove.6B.zip --no-check-certificate && unzip glove.6B.zip
# RUN python -m nltk.downloader punkt

CMD zsh
