import os
import random
import argparse

parser = argparse.ArgumentParser(description="Parser for all training options in RAF")
parser.add_argument('--train_ratio',type=float,default=0.9)
parser.add_argument('--txt_source',type=str,default='../DataSet/RAF/compound/EmoLabel/list_patition_label.txt')
parser.add_argument('--dst_location',type=str,default='../DataSet/RAF/compound/')
args = parser.parse_args()

if __name__ == '__main__':
    lines = open(args.txt_source).readlines()

    train_set = open(os.path.join(args.dst_location,'train_set'),'w')
    test_set = open(os.path.join(args.dst_location,'test_set'),'w')

    for i in range(0,3162):
        train_set.write(lines[i].replace('.','_aligned.'))
    for i in range(3162,3954):
        test_set.write(lines[i].replace('.','_aligned.'))
