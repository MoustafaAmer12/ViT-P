#!/bin/bash

echo "*****Starting Setup*****"
mkdir -p ./tmp_run/datasets

cp -r ./ViT-P/dinov2 ./tmp_run

cp -r ./ViT-P/datasets/cityscapes ./tmp_run/datasets
echo "*****Images are ready*****"


. venv/bin/activate
echo "*****Virtual Environment Activated*****"

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "*****PYTHONPATH is set*****"

cd ./tmp_run
echo "*****Changed Directory*****"

echo "*****Starting Training*****"
python3 ./dinov2/train/train.py \
  --config-file dinov2/configs/OneFormer/vitb14_Cityscapes.yaml \
  --output-dir ./OUTPUT_DIR \
  --no-resume
echo "*****Training Finished*****"

tar -cf ~/projects/model_cityscapes_base.tar checkpoint.pth
echo "*****Model Archived*****"