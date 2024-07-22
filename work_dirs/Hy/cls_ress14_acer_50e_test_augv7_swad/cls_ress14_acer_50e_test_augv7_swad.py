train_swad = dict(n_tolerance=32, tolerance_ratio=0.3, only_swad=False)
model = dict(
    type='Classifier',
    encoder=dict(type='resnest14d', pretrained=False, num_classes=2),
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(w_cls=1.0))
data_root = './'
img_norm_cfg = dict(
    mean=[
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5
    ],
    std=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0
    ],
    to_rgb=False)
data = dict(
    train=dict(
        type='HySpeFASDataset',
        data_root='/root/work/code/HySpe/data/hy',
        ann_files=['train.txt', 'val_gt.txt'],
        img_prefix=['/root/work/data/HySpeFAS_trainval/images'],
        test_mode=False,
        pipeline=dict(
            CoarseDropout=dict(
                max_holes=16,
                max_height=16,
                max_width=16,
                min_holes=8,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.1),
            RandomFlip=dict(hflip_ratio=0.5, vflip_ratio=0),
            RandomRotate=dict(max_angle=8, rotate_ratio=0.5),
            RandomCrop=dict(crop_ratio=0.3, crop_range=(0.1, 0.1)),
            RandomBorderMask=dict(vmask_range=[0.5, 1], ratio=0.3),
            Resize=dict(scale=(112, 112)),
            Normalize=dict(
                mean=[
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                std=[
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                ],
                to_rgb=False))),
    val=dict(
        type='HySpeFASDataset',
        data_root='/root/work/code/HySpe/data/hy',
        ann_files=['val_gt.txt'],
        img_prefix=['/root/work/data/HySpeFAS_trainval/images'],
        test_mode=True,
        pipeline=dict(
            Resize=dict(scale=(112, 112)),
            Normalize=dict(
                mean=[
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                std=[
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                ],
                to_rgb=False))),
    test=dict(
        type='HySpeFASDataset',
        data_root='/root/work/code/HySpe/data/hy',
        ann_files=['test.txt'],
        img_prefix=['/root/work/data/HySpeFAS_test/images'],
        test_mode=True,
        pipeline=dict(
            Resize=dict(scale=(112, 112)),
            Normalize=dict(
                mean=[
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                std=[
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                ],
                to_rgb=False))),
    train_loader=dict(
        num_gpus=1, shuffle=True, samples_per_gpu=16, workers_per_gpu=4),
    test_loader=dict(
        num_gpus=1, shuffle=False, samples_per_gpu=16, workers_per_gpu=4))
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=dict(
        loss_types=['loss'], eval_types=['apcer', 'bpcer', 'acer', 'auc']))
eval_cfg = dict(
    interval=100,
    score_type='acer',
    tsne_cfg=dict(marks=None, filename='tsne.png'))
optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0)
sched_cfg = dict(type='CosineLR', gamma=0.01, warmup=500.0, total_epochs=50)
check_cfg = dict(
    interval=50000.0,
    iters=12100,
    save_topk=3,
    load_from='work_dirs/Hy/cls_ress14_acer_50e_test_augv7_swad/latest.pth',
    resume_from=None,
    pretrain_from=None)
total_epochs = 50
freeze_cfg = None
work_dir = 'work_dirs/Hy/cls_ress14_acer_50e_test_augv7_swad'
gpu_ids = range(0, 1)
seed = 10
