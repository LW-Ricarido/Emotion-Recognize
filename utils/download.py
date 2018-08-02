from urllib import request
import os
import argparse

parser = argparse.ArgumentParser(description="Parser for all training options in RAF")
parser.add_argument('--train_ratio',type=float,default=0.9)
parser.add_argument('--txt_source',type=str,default='../DataSet/EmotioNet/download_set')
parser.add_argument('--dst_location',type=str,default='../DataSet/EmotioNet')
args = parser.parse_args()

lines = open(args.txt_source).readlines()

for i in range(5):
    print(i)
    name = str(i)+'.jpg'
    request.urlretrieve(lines[i].split(' ')[0],os.path.join(args.dst_location,name))