# -*- coding: UTF-8 -*-


from .ds_loss import DSLoss
from .l1_loss import L1Loss, SmoothL1Loss
from .kld_loss import KLDLoss, BKLDLoss
from .focal_loss import FocalLoss
from .mmd_loss import MMDLoss
from .triplet_loss import TripletLoss, AsymTripletLoss
from .onecenter_loss import OneCenterLoss

__all__ = [
    'DSLoss', 'L1Loss', 'KLDLoss', 'BKLDLoss', 'SmoothL1Loss', 'FocalLoss', 'MMDLoss',
    'TripletLoss', 'AsymTripletLoss'
]
