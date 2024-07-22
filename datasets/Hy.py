# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

import os
import cv2
import copy
import warnings
import scipy.io
import numpy as np
from utils.fileio import load
from utils.cv_util import distance_pt
from datasets.custom import CustomDataset

class HySpeFASDataset(CustomDataset):
    """HySpeFASDataset dataset for FAS.
    The annotation format is show as follows:
        -- annotation.txt
            ...
            img_path/img_name.mat label
            ...
    Args:
        ann_file (str or list[str]): Annotation file path
        pipeline (dict): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``mask_prefix``,  if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    NAME = "HySpeFASDataset"

    def __init__(self,
                 data_root,
                 ann_files,
                 pipeline=None,
                 img_prefix='',
                 test_mode=False,
                 ):
        super(HySpeFASDataset, self).__init__(
            data_root,
            ann_files,
            pipeline,
            img_prefix,
            test_mode)

    def load_annotations(self, ann_files):
        """load annotation information"""
        ann_nums = dict()
        _labels = list()
        _filenames = list()
        for i, ann_file in enumerate(ann_files):
            lines, thr = self._parse_thr_from_filename(ann_file)
            ann_nums[os.path.splitext(os.path.basename(ann_file))[0]] = len(lines)
            for line in lines:
                try:
                    label = int(line[-1])
                except:
                    label = 1
                # label  = 1 if label > 1 and self.test_mode else 0
                _labels.append(label)
                _filenames.append(os.path.join(self.img_prefix[i], line[0]))
 
        self.labels = np.array(_labels)
        self.filenames = np.array(_filenames)
        self.ann_nums = ann_nums

    def _parse_thr_from_filename(self, ann_file):
        thr = [-1, -1]
        if 'pseudo_' in ann_file:
            thr = [0.1, 0.9]
            data = os.path.splitext(os.path.basename(ann_file))[0].split('_')
            if len(data) == 4:
                ann_file = os.path.join(os.path.dirname(ann_file), f'pseudo_{data[1]}.txt')
                thr = [float(f'0.{data[2]}'), float(f'0.{data[3]}')]
            print('{} Neg thr: {}, Pos thr: {}'.format(os.path.basename(ann_file), thr[0], thr[1]))
        lines = load(ann_file)

        return lines, thr

    def _get_ann_info(self, filename, label):
        """read images and annotations"""
        img = scipy.io.loadmat(filename)['var']
        data = dict(
            img=img,
            label=np.ones((1,)).astype(np.int64) * label,
            path=filename)

        return data

    def __getitem__(self, idx):
        while True:
            label = copy.deepcopy(self.labels[idx])
            filename = copy.deepcopy(self.filenames[idx])
            # try:
            data = self._get_ann_info(filename, label)
            # except:
                # warnings.warn('Fail to read image: {}'.format(filename))
                # idx = self._rand_another(idx)
                # continue
            break
        data = self.pipeline(data)
        return data