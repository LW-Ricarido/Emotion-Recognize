import os
import random
import argparse

parser = argparse.ArgumentParser(description="Parser for all training options in RAF")
parser.add_argument('--train_ratio',type=float,default=0.9)
parser.add_argument('--txt_source',type=str,default='../DataSet/RAF/basic/EmoLabel/list_patition_label.txt')
parser.add_argument('--dst_location',type=str,default='../DataSet/RAF/basic/')
args = parser.parse_args()

if __name__ == '__main__':
    lines = open(args.txt_source).readlines()

    train_set = open(os.path.join(args.dst_location,'train.test'),'w')
    validation_set = open(os.path.join(args.dst_location,'validation_set'),'w')
    test_set = open(os.path.join(args.dst_location,'test_set'),'w')

    for i in range(0,12271):
        if random.random() <= args.train_ratio:
            train_set.write(lines[i])
        else:
            validation_set.write(lines[i])
    for i in range(12271,15339):
        test_set.write(lines[i])
