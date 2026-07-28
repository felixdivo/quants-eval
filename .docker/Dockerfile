# syntax = docker/dockerfile:1.2
ARG KIND=3.11
FROM mkrausio/ml_research:base-3.11-cuda-12.1.0

ARG COMPOSE_PROJ_NAME="quants-eval"
ARG GITHUB_TOKEN
ARG CUDA_VERSION=12.1.0

ENV GITHUB_TOKEN=${GITHUB_TOKEN}

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt autoremove python3 -y &&\
  rm -rf /usr/bin/python3 /usr/bin/pip3 /bin/python3 /bin/pip3 &&\
  ln -s /usr/bin/python${KIND} /usr/bin/python3 &&\
  ln -s /usr/bin/pip${KIND} /usr/bin/pip3 

RUN apt update && apt install git-lfs python3.11-dev libnuma-dev -y && git lfs install

RUN mkdir -p workspaces/$PROJECT_NAME

WORKDIR /workspaces/$PROJECT_NAME

ENV HF_HOME=/workspaces/$PROJECT_NAME/.cache/huggingface

COPY requirements.txt .

RUN ldconfig /usr/local/cuda-12.1/compat/

RUN --mount=type=cache,target=/root/.cache/pip \
  python -m pip install --upgrade pip && \
  pip install "torch==2.4.0" -r requirements.txt && \
  pip install sglang[all]

CMD zsh
