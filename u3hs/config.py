# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config


def add_u3hs_config(cfg):
    """
    Add config for U3HS.
    """
    # Reuse DeepLab config.
    add_deeplab_config(cfg)
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # Panoptic-DeepLab instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "U3HSInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
    cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    # U3HS post-processing setting.
    cfg.MODEL.U3HS = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.U3HS.STUFF_AREA = 2048

    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.U3HS.PREDICT_INSTANCES = True
    cfg.MODEL.U3HS.USE_DEPTHWISE_SEPARABLE_CONV = False
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    cfg.MODEL.U3HS.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.U3HS.BENCHMARK_NETWORK_SPEED = False
    # Embed head config
    cfg.MODEL.EMBED_HEAD = CN()
    cfg.MODEL.EMBED_HEAD.NAME = "U3HSEmbedHead"
    cfg.MODEL.EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.EMBED_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.EMBED_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.EMBED_HEAD.NUM_STUFF = 11
    cfg.MODEL.EMBED_HEAD.EMBEDDING_DIM = 8
    cfg.MODEL.EMBED_HEAD.NUM_CLASSES = 19
    cfg.MODEL.EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.EMBED_HEAD.CENTER_THRESHOLD = 0.1
    cfg.MODEL.EMBED_HEAD.NMS_KERNEL = 7
    cfg.MODEL.EMBED_HEAD.TOP_K_INSTANCE = 200
