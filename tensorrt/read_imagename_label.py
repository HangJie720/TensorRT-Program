# -*- coding: utf-8 -*-

DATASET_DIR="/ILSVRC2012/raw-data/imagenet-data/validation"
import os
import numpy as np
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件


def GetOrderDirList(dir, fileList, label):

    if os.path.isfile(dir):
        fileList.append(dir.split('/')[-1]+' '+str(label))
    elif os.path.isdir(dir):
        for s in sorted(os.listdir(dir)):
            newDir = os.path.join(dir, s)
            for p in sorted(os.listdir(newDir)):
                fileList.append(p.split('/')[-1]+' '+str(label))
            label += 1

    return fileList


file_list = GetOrderDirList(DATASET_DIR, [], 1)
with open("val_new.txt", "w") as f:
    for list in file_list:
        f.write(list)
        f.write("\n")
