import os
import random
import argparse
import csv

parser = argparse.ArgumentParser(description="Parser for all training options in RAF")
parser.add_argument('--train_ratio',type=float,default=0.9)
parser.add_argument('--txt_source',type=str,default='../DataSet/EmotioNet/URLsWithEmotionCat.csv')
parser.add_argument('--dst_location',type=str,default='../DataSet/EmotioNet')
args = parser.parse_args()

if __name__ == '__main__':
    lines = open(args.txt_source)
    lines = csv.reader(lines)

    train_set = open(os.path.join(args.dst_location,'train_set'),'w')
    test_set = open(os.path.join(args.dst_location,'test_set'),'w')
    download_set = open(os.path.join(args.dst_location,'download_set'),'w')

    for line in lines:
        for j in range(16):
            if line[j+2] == '1':
                label = str(j)
                tmp = line[0] + ' '+ label + '\n'
                download_set.write(tmp)
                break
    download_set.close()
