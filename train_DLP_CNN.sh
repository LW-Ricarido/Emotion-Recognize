#!/usr/bin/env bash
mkdir checkpoints

python3 -u main.py \
    -shuffle \
    -train_record \
    -model DLP_CNN \
    -data_dir DataSet/RAF/basic/Image/aligned/ \
    -train_list DataSet/RAF/basic/train_set \
    -test_list DataSet/RAF/basic/validation_set \
    -save_path checkpoints \
    -output_classes 7 \
    -n_epochs 40 \
    -learn_rate 0.01 \
    -batch_size 64 \
    -k 3 \
    -lam 5 \
    -criterion DLP_LOSS \
    -workers 2 \
    -nGPU 1 \
    -decay 30 \
    -size 100 \
2>&1 | tee train_DLP_CNN.log
