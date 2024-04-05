# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm, Linear
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.structures import ImageList, Instances
from detectron2.utils.registry import Registry

from .utils.loss import DiscriminativeLoss
from .post_processing import get_panoptic_segmentation

__all__ = ["U3HS", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]

INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class U3HS(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.embed_head = build_embed_branch(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.U3HS.STUFF_AREA
        self.predict_instances = cfg.MODEL.U3HS.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = (
            cfg.MODEL.U3HS.USE_DEPTHWISE_SEPARABLE_CONV
        )
        assert (
                cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
                == cfg.MODEL.U3HS.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.U3HS.SIZE_DIVISIBILITY
        self.benchmark_network_speed = (
            cfg.MODEL.U3HS.BENCHMARK_NETWORK_SPEED
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)

        features = self.backbone(images.tensor)

        losses = {}
        if "sem_seg" in batched_inputs[0]:
            semantic_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            semantic_targets = ImageList.from_tensors(
                semantic_targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            semantic_targets = None
            weights = None

        semantic_logits, uncertainties, sem_seg_losses = self.sem_seg_head(
            features, semantic_targets, weights
        )
        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

        center_results, detection_feature, center_losses = self.ins_embed_head(
            features, center_targets, center_weights, semantic_logits=semantic_logits
        )
        losses.update(center_losses)

        if "center_points" in batched_inputs[0]:
            center_points = [x["center_points"] for x in batched_inputs]
        else:
            center_points = None

        if "panoptic" in batched_inputs[0]:
            panoptic_targets = [x["panoptic"].to(self.device) for x in batched_inputs]
            panoptic_targets = ImageList.from_tensors(panoptic_targets, size_divisibility).tensor
        else:
            panoptic_targets = None

        if "panoptic_ids" in batched_inputs[0]:
            panoptic_ids = [x["panoptic_ids"] for x in batched_inputs]
        else:
            panoptic_ids = None

        (
            embeddings,
            association_scores,
            total_centers,
            prototype_loss,
            discriminative_loss,
        ) = self.embed_head(
            features,
            center_points,
            semantic_targets,
            panoptic_targets,
            panoptic_ids,
            semantic_logits=semantic_logits,
            detection_feature=detection_feature,
            center_heatmap=center_results,
        )
        losses.update(prototype_loss)
        losses.update(discriminative_loss)

        if self.training:
            return losses
        if self.benchmark_network_speed:
            return []
        processed_results = []
        for (
                semantic_logit,
                embedding,
                uncertainty,
                total_center,
                association_score,
                input_per_image,
                image_size,
        ) in zip(
            semantic_logits,
            embeddings,
            uncertainties,
            total_centers,
            association_scores,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(semantic_logit, image_size, height, width)
            e = sem_seg_postprocess(embedding, image_size, height, width)
            u = sem_seg_postprocess(uncertainty, image_size, height, width)
            a = sem_seg_postprocess(association_score, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, sem_seg = get_panoptic_segmentation(
                total_center,
                r,
                e,
                u,
                a,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
            )
            processed_results.append({"sem_seg": sem_seg})
            panoptic_image = panoptic_image.squeeze(0)
            # processed_results.append({"panoptic_unknown": panoptic_image})

            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        u_squeezed = u.squeeze(0)
                        uncertainty_scores = torch.mean(u_squeezed[mask])
                        instance.scores = torch.tensor(
                            [uncertainty_scores], device=panoptic_image.device
                        )
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class U3HSSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            loss_weight: float,
            ignore_value: int,
            num_classes: int,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        This method is the forward pass of the U3HSSemSegHead class. It computes the semantic logits and evidence,
        and returns the semantic logits, uncertainties, and losses if in training mode.

        Args:
            features (torch.Tensor): The input features to the head.
            targets (torch.Tensor, optional): The ground truth targets. Default is None.
            weights (torch.Tensor, optional): The weights for the targets. Default is None.

        Returns:
            In training mode:
                semantic_logits (torch.Tensor): The semantic logits. (B,19,H,W)
                None: No uncertainties are returned in training mode.
                losses (dict): The losses computed by the losses method.

            In inference mode:
                semantic_logits (torch.Tensor): The semantic logits. (B,19,H,W)
                uncertainties (torch.Tensor): The uncertainties computed as 19 divided by the strength.
                {} (dict): An empty dictionary.
        """
        semantic_logits, evidence = self.layers(features)
        alpha = evidence + 1
        strength = torch.sum(alpha, dim=1, keepdim=True)
        if self.training:
            return semantic_logits, None, self.losses(alpha, targets, weights)
        else:
            uncertainties = 19 / strength
            return semantic_logits, uncertainties, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        semantic_logits = y
        evidence = F.softplus(y)
        return semantic_logits, evidence

    def losses(self, alpha, targets, weights=None):
        alpha_upsampled = F.interpolate(
            alpha, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        num_classes = alpha_upsampled.size(1)
        batch_size, H, W = targets.size()
        ignore_mask = (targets != 255).unsqueeze(1)
        targets_copy = targets.clone()
        targets_copy[targets_copy == 255] = 0
        target_one_hot = torch.zeros(batch_size, num_classes, H, W, device=alpha.device)
        target_one_hot.scatter_(1, targets_copy.unsqueeze(1), 1)
        strength = alpha_upsampled.sum(dim=1, keepdim=True)
        loss = (
                target_one_hot * (torch.digamma(strength) - torch.digamma(alpha_upsampled))
        ).sum(dim=1)
        loss = loss * ignore_mask.squeeze(1)

        if weights is not None:
            normalization_factor = (ignore_mask * weights).sum()
            loss = (loss * weights).sum() / normalization_factor
        else:
            loss = loss.sum() / ignore_mask.sum()

        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class U3HSInsEmbedHead(DeepLabV3PlusHead):
    """
    AN instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            norm: Union[str, Callable],
            head_channels: int,
            center_loss_weight: float,
            num_classes: int,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
        """
        super().__init__(
            input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs
        )
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels + num_classes, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm_semantic = get_norm(norm, num_classes)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
                len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        return ret

    def forward(
            self,
            features,
            center_targets=None,
            center_weights=None,
            semantic_logits=None,
    ):
        """
        This method is the forward pass of the U3HSInsEmbedHead class. It computes the center prediction
        and returns the center prediction, detection features, and center losses if in training mode.

        Args:
            features (dict[torch.tensor]): The input features to the head.
            center_targets (torch.Tensor, optional): The ground truth center targets. Default is None. (B,1,H',W')
            center_weights (torch.Tensor, optional): The weights for the center targets. Default is None. (B,1,H',W')
            semantic_logits (torch.Tensor, optional): The semantic logits from the semantic segmentation head. Default is None. (B,19,H,W)

        Returns:
            In training mode:
                None: No logits are returned in traicning mode.
                detection_feature (torch.Tensor): The output of DeepLabV3PlusHead. (B,128,H,W)
                center_losses (dict): The center losses computed by the center_losses method.

            In inference mode:
                center (torch.Tensor): The center prediction. (B,1,H,W)
                detection_feature (torch.Tensor): The output of DeepLabV3PlusHead. (B,128,H,W)
                {} (dict): An empty dictionary.
        """
        detection_feature = self.layers(features)
        # B, _, H, W = detection_feature.shape
        # layer_norm_semantic = nn.LayerNorm([19, H, W]).to(self.device)
        center = self.center_head(detection_feature)  # center_feature (B,32,128,256)
        if semantic_logits is not None:
            semantic_logits_detached = semantic_logits.detach()
            # semantic_logits_detached = layer_norm_semantic(semantic_logits_detached)
            semantic_logits_detached = self.norm_semantic(semantic_logits_detached)
            center = torch.cat([center, semantic_logits_detached], dim=1)

        center = self.center_predictor(center)

        if self.training:
            return (
                None,
                detection_feature,
                self.center_losses(center, center_targets, center_weights),
            )
        else:
            return center, detection_feature, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        return y

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses


def build_embed_branch(cfg, input_shape):
    if cfg.INPUT.CROP.ENABLED:
        assert cfg.INPUT.CROP.TYPE == "absolute"
        train_size = cfg.INPUT.CROP.SIZE
    else:
        train_size = None
    decoder_channels = [cfg.MODEL.EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.EMBED_HEAD.IN_FEATURES) - 1
    ) + [cfg.MODEL.EMBED_HEAD.ASPP_CHANNELS]

    return U3HSEmbedHead(
        input_shape=input_shape,
        project_channels=cfg.MODEL.EMBED_HEAD.PROJECT_CHANNELS,
        aspp_dilations=cfg.MODEL.EMBED_HEAD.ASPP_DILATIONS,
        aspp_dropout=cfg.MODEL.EMBED_HEAD.ASPP_DROPOUT,
        decoder_channels=decoder_channels,
        detection_branch_decoder_channels=cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM,
        common_stride=cfg.MODEL.EMBED_HEAD.COMMON_STRIDE,
        norm=cfg.MODEL.EMBED_HEAD.NORM,
        train_size=train_size,
        head_channels=cfg.MODEL.EMBED_HEAD.HEAD_CHANNELS,
        embedding_dim=cfg.MODEL.EMBED_HEAD.EMBEDDING_DIM,
        num_classes=cfg.MODEL.EMBED_HEAD.NUM_CLASSES,
        num_stuff=cfg.MODEL.EMBED_HEAD.NUM_STUFF,
        center_threshold=cfg.MODEL.EMBED_HEAD.CENTER_THRESHOLD,
        nms_kernel=cfg.MODEL.EMBED_HEAD.NMS_KERNEL,
        top_k_instance=cfg.MODEL.EMBED_HEAD.TOP_K_INSTANCE,
    )

class U3HSEmbedHead(DeepLabV3PlusHead):
    """
    AN instance embedding head described in :paper:`Segmenting Known Objects and Unseen Unknowns without Prior Knowledge`.
    """
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            decoder_channels: List[int],
            detection_branch_decoder_channels: int,
            norm: Union[str, Callable],
            head_channels: int,
            embedding_dim: int,
            num_classes: int,
            num_stuff: int,
            center_threshold: float,
            nms_kernel: int,
            top_k_instance: int,
            **kwargs,
    ):
        super().__init__(
            input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs
        )

        assert self.decoder_only
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.top_k_instance = top_k_instance
        self.stuff_num = num_stuff

        use_bias = norm == ""
        self.embedding_head = nn.Sequential(
            Conv2d(
                decoder_channels[0] + detection_branch_decoder_channels,
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
        )
        self.prototype_head = nn.Sequential(
            Conv2d(
                decoder_channels[0] + detection_branch_decoder_channels,
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.embedding_head[0])
        weight_init.c2_xavier_fill(self.embedding_head[1])
        weight_init.c2_xavier_fill(self.prototype_head[0])
        weight_init.c2_xavier_fill(self.prototype_head[1])
        self.embedding_conv = Conv2d(head_channels + num_classes, embedding_dim, kernel_size=1)
        nn.init.normal_(self.embedding_conv.weight, 0, 0.001)
        nn.init.constant_(self.embedding_conv.bias, 0)

        self.prototype_conv = Conv2d(head_channels + num_classes, embedding_dim, kernel_size=1)
        nn.init.normal_(self.prototype_conv.weight, 0, 0.001)
        nn.init.constant_(self.prototype_conv.bias, 0)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.stuff_prototype_layer = Linear(embedding_dim, num_stuff * (embedding_dim + 1))
        nn.init.normal_(self.stuff_prototype_layer.weight, 0, 0.001)
        nn.init.constant_(self.stuff_prototype_layer.bias, 0)

        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminative_loss = DiscriminativeLoss(delta_var=0.5,
                                                      delta_dist=1.5,
                                                      norm=2,
                                                      usegpu=torch.cuda.is_available())

        self.norm_semantic = get_norm(norm, num_classes)

    def forward(
            self,
            features,
            center_points,
            semantic_targets,
            panoptic_targets,
            panoptic_ids,
            semantic_logits=None,
            detection_feature=None,
            center_heatmap=None,
    ):
        """
           This method is the forward pass of the U3HSEmbedHead class. It computes the embeddings and prototypes,
           and returns the embeddings, association scores, center numbers, prototype losses, and discriminative losses if in training mode.

           Args:
               features (dict[torch.Tensor]): The input features to the head.
               center_points (List): The list of center points for each image in the batch.
               semantic_targets (torch.Tensor): The semantic segmentation ground truth. (B,H',W')
               panoptic_targets (torch.Tensor): The panoptic segmentation ground truth. (B,H',W')
               panoptic_ids (List): The list of panoptic ids for each image in the batch.
               semantic_logits (torch.Tensor, optional): The semantic logits from the semantic segmentation head. Default is None. (B,19,H,W)
               detection_feature (torch.Tensor, optional): The detection features from the instance embedding head. Default is None. (B,128,H,W)
               center_heatmap (torch.Tensor, optional): The center heatmap from the instance embedding head. Default is None.

           Returns:
               In training mode:
                   {} (dict): An empty dictionary.
                   {} (dict): An empty dictionary.
                   {} (dict): An empty dictionary.
                   prototype_loss (dict): The prototype loss computed by the compute_prototype_loss method.
                   discriminative_loss (dict): The discriminative loss computed by the compute_discriminative_loss method.

               In inference mode:
                   embedding (torch.Tensor): The embeddings. (B,8,H,W)
                   association_score (torch.Tensor): The association scores. (B,self.top_k_instance+self.stuff_num,H,W)
                   center_num (torch.Tensor): The number of centers.
                   {} (dict): An empty dictionary.
                   {} (dict): An empty dictionary.
        """
        decoder_output = self.layers(features)

        B, _, H, W = decoder_output.shape
        # layer_norm_semantic = nn.LayerNorm([19, H, W]).to(self.device)
        # Instance aware: concatenate detection branch features to prototypes and embeddings heads
        if detection_feature is not None:
            detection_feature_detached = detection_feature.detach()
            # detection_feature_detached = self.batch_norm_detection(detection_feature_detached)
            decoder_output = torch.cat([decoder_output, detection_feature_detached], dim=1)

        embedding_feature = self.embedding_head(decoder_output)
        prototype_feature = self.prototype_head(decoder_output)

        # Semantic aware: concatenate semantic logits to the last layers of heads
        if semantic_logits is not None:
            semantic_logits_detached = semantic_logits.detach()
            # semantic_logits_detached = layer_norm_semantic(semantic_logits_detached)
            # semantic_logits_detached = self.batch_norm_semantic(semantic_logits_detached)
            semantic_logits_detached = self.norm_semantic(semantic_logits_detached)
            embedding_feature = torch.cat([embedding_feature, semantic_logits_detached], dim=1)
            prototype_feature = torch.cat([prototype_feature, semantic_logits_detached], dim=1)

        embedding = self.embedding_conv(embedding_feature)
        prototype_input = self.prototype_conv(prototype_feature)

        if self.training:
            center_points, dis_targets, n_objects, pro_targets, valid_mask = self.downsample_and_generate_targets(
                center_points, semantic_targets, panoptic_targets, panoptic_ids, H, W
            )
            padded_prototypes = self.compute_prototypes(prototype_input, center_points)  # Shape: (B, N, C + 1)
            # Compute association score for each pixel and prototype
            association_score = self.compute_association_scores(padded_prototypes, embedding)  # Shape: (B, N, H, W)
            prototype_loss = self.compute_prototype_loss(association_score, pro_targets, valid_mask)
            discriminative_loss = self.compute_discriminative_loss(embedding, dis_targets, n_objects)
            return {}, {}, {}, prototype_loss, discriminative_loss
        else:
            centers_count = []
            center_points = []
            # Find the center points from the center heatmap
            center_heatmap = F.threshold(center_heatmap, self.center_threshold, -1)
            for b in range(B):
                # NMS
                nms_padding = (self.nms_kernel - 1) // 2
                heatmap = center_heatmap[b:b + 1]
                heatmap_max_pooled = F.max_pool2d(heatmap, kernel_size=self.nms_kernel, stride=1, padding=nms_padding)
                heatmap[heatmap != heatmap_max_pooled] = -1
                # find topk centers
                top_k_scores, _ = torch.topk(heatmap.view(-1), self.top_k_instance)
                top_centers = torch.nonzero(heatmap.view(H, W) > top_k_scores[-1].clamp_(min=0), as_tuple=False)
                center_points.append(top_centers)
                centers_count.append(top_centers.shape[0])
                center_num = torch.tensor(centers_count)
            padded_prototypes = self.compute_prototypes(prototype_input, center_points)
            # Compute association score for each pixel and prototype
            association_score = self.compute_association_scores(padded_prototypes, embedding)  # Shape: (B, N, H, W)
            return embedding, association_score, center_num, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        return y

    def downsample_and_generate_targets(self, center_points, semantic_targets, panoptic_targets, panoptic_ids, h, w):
        B, original_h, original_w = semantic_targets.shape
        semantic_targets = semantic_targets.float().unsqueeze(1)
        semantic_targets = F.interpolate(semantic_targets, size=(h, w), mode="nearest").squeeze(1).long()
        panoptic_targets = panoptic_targets.float().unsqueeze(1)
        panoptic_targets = F.interpolate(panoptic_targets, size=(h, w), mode="nearest").squeeze(1).long()
        scale_y = h / original_h
        scale_x = w / original_w
        center_points_downsampled = []
        max_n_objects = self.top_k_instance + self.stuff_num
        # Targets for discriminative loss
        dis_targets = torch.zeros((B, max_n_objects, h, w), dtype=torch.float32, device=self.device)
        # Targets for prototype loss
        pro_targets = torch.zeros((B, max_n_objects, h, w), dtype=torch.float32, device=self.device)
        valid_mask = torch.zeros(B, dtype=torch.uint8, device=self.device)
        n_objects = torch.zeros(B, dtype=torch.uint8, device=self.device)
        scaled_center_points = []
        for batch_center_points in center_points:
            batch_scaled_points = [[point[0] * scale_y, point[1] * scale_x] for point in batch_center_points]
            scaled_center_points.append(batch_scaled_points)
        for i in range(B):
            valid_idx_dis = 0  # Index for discriminative targets
            valid_idx_pro = 0  # Index for prototype targets
            valid_center_points = []
            existing_ids = panoptic_targets[i].unique()
            for panoptic_id, point in zip(panoptic_ids[i], scaled_center_points[i]):
                if (panoptic_id == existing_ids).any():
                    center_mask = (panoptic_targets[i, :, :] == panoptic_id)
                    if center_mask.any():
                        dis_targets[i, valid_idx_dis, :, :] = center_mask
                        pro_targets[i, valid_idx_pro, :, :] = center_mask
                        valid_center_points.append(point)
                        valid_idx_dis += 1
                        valid_idx_pro += 1
            for class_id in range(self.stuff_num):
                stuff_mask = (semantic_targets[i, :, :] == class_id)
                if stuff_mask.any():
                    dis_targets[i, valid_idx_dis, :, :] = stuff_mask
                    valid_idx_dis += 1
                pro_targets[i, valid_idx_pro, :, :] = stuff_mask
                valid_idx_pro += 1
            n_objects[i] = valid_idx_dis
            valid_mask[i] = valid_idx_pro
            if valid_center_points:
                center_points_downsampled.append(
                    torch.tensor(valid_center_points, dtype=torch.float32, device=self.device))
            else:
                center_points_downsampled.append(torch.empty(0, 2, device=self.device))

        return center_points_downsampled, dis_targets, n_objects, pro_targets, valid_mask


    def compute_prototypes(self, prototype_input, center_points):
        B, C, H, W = prototype_input.shape
        max_n_objects = self.top_k_instance + self.stuff_num
        padded_prototypes = torch.zeros((B, max_n_objects, C + 1), device=self.device)

        offsets = torch.tensor([[0, 0], [-1, 1], [1, 1], [1, -1], [-1, -1]], device=self.device)
        distances = torch.sqrt(torch.sum(offsets ** 2, dim=1, keepdim=True))
        distances[0] = 1
        weights = 1 / distances
        weights /= weights.sum()

        pooled_feature = self.global_avg_pool(prototype_input).view(B, -1)
        stuff_prototypes = self.stuff_prototype_layer(pooled_feature).view(-1, self.stuff_num, self.embedding_dim + 1)
        for b in range(B):
            centers = center_points[b]
            if len(centers) == 0:
                continue
            for center in centers:
                y, x = center.long()
                neighbors_y = (y + offsets[:, 0]).clamp(0, H - 1)
                neighbors_x = (x + offsets[:, 1]).clamp(0, W - 1)
                neighbor_features = prototype_input[b, :, neighbors_y, neighbors_x]

                weighted_features = neighbor_features * weights.T
                weighted_mean = weighted_features.sum(dim=-1) / weights.sum()

                diffs = neighbor_features - weighted_mean.unsqueeze(-1)
                weighted_var = (diffs ** 2 * weights.T).sum(dim=-1) / weights.sum()
                var_mean = weighted_var.mean().unsqueeze(0)

                index = (padded_prototypes[b, :, 0] != 0).sum().item()
                padded_prototypes[b, index, :-1] = weighted_mean
                padded_prototypes[b, index, -1] = var_mean

            start_index = (padded_prototypes[b, :, 0] != 0).sum().item()
            padded_prototypes[b, start_index:start_index + self.stuff_num, :] = stuff_prototypes[b]

        return padded_prototypes

    def compute_association_scores(self, prototypes, embeddings):
        B, _, H, W = embeddings.shape
        means = prototypes[:, :, :-1]  # Shape: (B, self.top_k_instance+self.stuff_num, 8)
        # Keep prototype variance positive by using softplus
        variance = F.softplus(prototypes[:, :, -1]).unsqueeze(-1)  # Shape: (B, self.top_k_instance+self.stuff_num, 1)
        # variance = torch.clamp(variance, min=1e-5)
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # Shape: (B, H*W, 8)
        means = means.unsqueeze(1)  # Shape: (B, 1, self.top_k_instance+self.stuff_num, 8)
        embeddings = embeddings.unsqueeze(2)  # Shape: (B, H*W, 1, 8)
        squared_l2_distance = ((means - embeddings) ** 2).sum(dim=-1)  # Shape: (B, H*W, self.top_k_instance+self.stuff_num)
        variance = variance.view(B, 1, -1)  # Shape: (B, 1, self.top_k_instance+self.stuff_num)
        association_scores = (-squared_l2_distance / (2 * variance)).view(B, H, W, -1).permute(0, 3, 1, 2)

        return association_scores

    def compute_prototype_loss(self, association_scores, targets, valid_mask):
        B, _, _, _ = association_scores.shape
        single_channel_targets = targets.argmax(dim=1)
        prototype_loss = 0
        for i in range(B):
            n_valid_channels = valid_mask[i].item()
            valid_association_scores = association_scores[i, :n_valid_channels, :, :]
            prototype_loss += F.cross_entropy(
                valid_association_scores.unsqueeze(0),
                single_channel_targets[i].unsqueeze(0),
                reduction="mean"
            )
        prototype_loss /= B
        losses = {"loss_prototype": prototype_loss}
        return losses

    def compute_discriminative_loss(self, embedding, targets, n_objects):
        discriminative_loss = self.discriminative_loss(embedding, targets, n_objects)
        losses = {"loss_discriminative": discriminative_loss}
        return losses
