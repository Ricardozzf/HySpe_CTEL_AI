# -*- coding: UTF-8 -*-
# !/usr/bin/env python3


import cv2

def distance_pt(pt1, pt2):
    return ((pt2[1] - pt1[1])**2 +  (pt2[0] - pt1[0])**2)**0.5