import os
import random
import argparse
import csv

parser = argparse.ArgumentParser(description="Parser for all training options in RAF")
parser.add_argument('--train_ratio',type=float,default=0.9)
parser.add_argument('--txt_source',type=str,default='../DataSet/EmotioNet/download_set')
parser.add_argument('--dst_location',type=str,default='../DataSet/EmotioNet')
args = parser.parse_args()

if __name__ == '__main__':
    lines = open(args.txt_source).readlines()

    train_set = open(os.path.join(args.dst_location,'train_set'),'w')
    val_set = open(os.path.join(args.dst_location,'val_set'),'w')
    test_set = open(os.path.join(args.dst_location,'test_set'),'w')

    for i in range(0,1381):
        tmp = str(i) + '.jpg ' + lines[i].split(' ')[1]
        t = random.random()
        if t < 0.8:
            train_set.write(tmp)
        elif t < 0.9:
            val_set.write(tmp)
        else:
            test_set.write(tmp)
    for i in range(1383,len(lines)):
        tmp = str(i) + '.jpg ' + lines[i].split(' ')[1]
        t = random.random()
        if t < 0.8:
            train_set.write(tmp)
        elif t < 0.9:
            val_set.write(tmp)
        else:
            test_set.write(tmp)
    train_set.close()
    test_set.close()


