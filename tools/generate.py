# -*- coding: UTF-8 -*-
# !/usr/bin/env python3
################################################################################
#
# Copyright (c) 2022 Chinatelecom.cn, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了 Metric

Authors: zouzhaofan(zouzhf41@chinatelecom.cn)
Date:    2022/08/22 19:05:08
"""

import os
import cv2
import sys
import math
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from sklearn.metrics import roc_curve, auc

def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Infer onnx')
    parser.add_argument('--prefix', default='work_dirs/Hy')
    parser.add_argument('--res', default=["cls_ress14_acer_50e_test_augv7_swad/score.txt",
                                          "cls_ress14_acer_50e_test_augv7_dropc_swad/score.txt"], nargs='+', help='the dir to save logs and models')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    dst_path = os.path.splitext(__file__)[0]
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    lines = []
    scores, filenames = [], []
    
    for re in args.res:
        with open(os.path.join(args.prefix, re), 'r') as f:
            lines.append(f.readlines())
  
    for line in zip(*lines):
        filenames.append(line[0].strip().split(' ')[0])
        score = [float(x.strip().split(' ')[-1]) for x in line]
        scores.append(np.array(score).mean())

    thr = 0.1592
    fout = open(os.path.join(dst_path,"result.txt"),"w")
    for name, score in zip(filenames, scores):
        label = 1 if score>thr else 0
        info = f"{os.path.basename(name)} {label}\n"
        fout.write(info)
    fout.close()
        
 

    


