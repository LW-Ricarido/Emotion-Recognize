#!/usr/bin/env bash

mkdir checkpoints/mixup

python3 -u main.py \
    -shuffle \
    -train_record \
    -model resnet50 \
    -data_dir DataSet/RAF/basic/Image/aligned/ \
    -train_list DataSet/RAF/basic/train_set \
    -test_list DataSet/RAF/basic/validation_set \
    -save_path checkpoints/pure \
    -output_classes 20 \
    -n_epochs 120 \
    -learn_rate 0.0003 \
    -criterion DLP_LOSS \
    -lam 0.05 \
    -k 10 \
    -batch_size 32 \
    -workers 0 \
    -nGPU 0 \
    -decay 30 \
2>&1 | tee train_resnet50_pure.log
