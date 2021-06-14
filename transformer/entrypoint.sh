#!/bin/bash

if [ ! -d /music/models ]; then
  echo "Downloading pretrained models..."
  mkdir /music/models
  gsutil -q -m cp -r gs://magentadata/models/music_transformer/checkpoints/* /music/models
  echo "Pretrained models downloaded to /music/models"
else
  echo "Model directory /music/models exists, skipping download"
fi

python /music/api.py