#!/bin/sh

#  test_DLP_CNN.sh
#  
#
#  Created by 李威 on 2018/7/23.
#  

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model DLP_CNN \
    -data_dir ../DataSet/RAF/basic/Image/aligned/ \
    -train_list ..DataSet/RAF/basic/train_set \
    -test_list ../DataSet/RAF/basic/validation_set \
    -save_path checkpoints \
    -output_classes 7 \
    -n_epochs 20 \
    -learn_rate 0.003 \
    -batch_size 64 \
    -workers 0 \
    -nGPU 1 \
    -decay 30 \
    -size 100 \
    -save_result \
    -test_only \
    -ckpt 30 \
    -resume \
2>&1 | tee test_DLP_CNN.log
