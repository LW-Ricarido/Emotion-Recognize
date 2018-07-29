#!/bin/sh

#  test_resnet50.sh
#  
#
#  Created by 李威 on 2018/7/29.
#  
python -u main.py \
    -shuffle \
    -train_record \
    -model resnet50 \
    -data_dir ../DataSet/RAF/basic/Image/aligned/ \
    -train_list ../DataSet/RAF/basic/train_set \
    -test_list ../DataSet/RAF/basic/test_set \
    -save_path checkpoints/pure \
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
    -ckpt 36 \
    -resume \
2>&1 | tee test_resnet50.log
