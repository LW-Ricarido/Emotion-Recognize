#!/usr/bin/env bash

mkdir checkpoints/mixup

python -u main.py \
    -shuffle \
    -train_record \
    -model resnet50 \
    -data_dir ./data/image_scene_training/data \
    -train_list ./data/image_scene_training/train.txt \
    -test_list ./data/image_scene_training/test.txt \
    -save_path checkpoints/pure \
    -output_classes 20 \
    -n_epochs 120 \
    -learn_rate 0.0003 \
    -pretrained ./pretrained/resnet50-19c8e357.pth \
    -batch_size 32 \
    -workers 8 \
    -nGPU 1 \
    -decay 30 \
2>&1 | tee train_resnet50_pure.log
