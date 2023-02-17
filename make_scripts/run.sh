#!/bin/bash

echo "Run container"
if [[ $USER == ml-* ]]; then
CUDA=1
else
CUDA=0
fi
((CUDA)) && echo "Enabled CUDA mode"
((!CUDA)) && echo "Enabled CPU mode"

COMPOSE_FOLDER=$(echo "-f .docker/compose.yml" && ((CUDA)) && echo "-f .docker/compose.cuda.yml")

docker compose $COMPOSE_FOLDER up -d timeseries_vit
docker compose $COMPOSE_FOLDER exec timeseries_vit zsh