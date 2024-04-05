from collections import Counter
import torch
import numpy as np
from sklearn.cluster import DBSCAN

def get_instance_segmentation(
    sem_seg,
    total_center,
    valid_thing_association_score,
    thing_seg,
    embedding,
    uncertainty,
):
    """
    This function generates instance segmentation from semantic segmentation and association score.

    Args:
        sem_seg (Tensor): Semantic segmentation tensor of shape [1, H, W].
        total_center (int): The total number of detected centers.
        valid_thing_association_score (Tensor): Tensor of shape [total_center, H, W] representing valid thing association scores.
        thing_seg (Tensor): Tensor of shape [1, H, W] representing thing segmentation.
        embedding (Tensor): Tensor of shape [embedding_dim, H, W] representing embeddings.
        uncertainty (Tensor): Tensor of shape [1, H, W] representing uncertainties.

    Returns:
        final_ins_seg (Tensor): Instance segmentation tensor.
        DBSCAN_run (bool): Boolean indicating if DBSCAN was run.
        outliers_mask (Tensor): Tensor indicating outliers.
    """
    device = sem_seg.device
    if total_center == 0:
        ins_seg = torch.zeros_like(sem_seg)
    else:
        # reserve id=0 for stuff
        ins_seg = torch.argmax(valid_thing_association_score, dim=0, keepdim=True) + 1
        ins_seg = thing_seg * ins_seg

    t = 3
    min_samples = 40
    eps = 0.3

    mean = torch.mean(uncertainty)
    std_variance = torch.std(uncertainty)
    threshold = mean + t * std_variance

    # Create a mask for the unknown pixels
    unknown_mask = uncertainty >= threshold

    num_unknown_pixels = torch.sum(unknown_mask)
    unknown_mask = unknown_mask.squeeze(0)

    embedding_np = embedding.cpu().detach().numpy().transpose(1, 2, 0)
    unknown_pixels = embedding_np[unknown_mask.cpu().numpy()]
    if num_unknown_pixels >= min_samples:
        DBSCAN_run = True
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(unknown_pixels)
        labels = dbscan.labels_
    else:
        DBSCAN_run = False
        labels = []

    ins_seg_np = ins_seg.cpu().numpy().squeeze(0)
    outliers_mask = np.zeros_like(ins_seg_np, dtype=bool)
    cluster_label = np.max(ins_seg_np) + 1
    unknown_indices = np.where(unknown_mask.cpu().numpy())

    for idx, label in enumerate(labels):
        if label != -1:
            ins_seg_np[unknown_indices[0][idx], unknown_indices[1][idx]] = label + cluster_label
        else:
            outliers_mask[unknown_indices[0][idx], unknown_indices[1][idx]] = True

    final_ins_seg = torch.from_numpy(ins_seg_np).to(ins_seg.dtype).to(device).unsqueeze(0)
    outliers_mask = torch.from_numpy(outliers_mask).to(torch.bool).to(device).unsqueeze(0)

    return final_ins_seg, DBSCAN_run, outliers_mask

def get_semantic_segmentation(sem_logit, thing_seg, valid_stuff_association_score):
    """
    This function generates semantic segmentation from the semantic logits, thing segmentation, and valid stuff association scores.

    Args:
        sem_logit (Tensor): A Tensor of shape [19, H, W], predicted category id for each pixel. Only use predictions for things.
        thing_seg (Tensor): A Tensor of shape [1, H, W], mask for thing categories.
        valid_stuff_association_score (Tensor): A Tensor of shape [11, H, W] representing valid stuff association scores.

    Returns:
        sem_seg_result (Tensor): Semantic segmentation result as a Tensor of shape [1, H, W].
    """
    device = sem_logit.device
    H, W = sem_logit.shape[-2:]
    thing_mask = thing_seg  # thing mask 11-18
    stuff_mask = ~thing_seg  # stuff mask 0-10
    # Determine semantic labels for things
    thing_semantic = torch.argmax(sem_logit[11:], dim=0) + 11
    # Determine semantic labels for stuff based on association scores
    stuff_semantic = torch.argmax(valid_stuff_association_score, dim=0)
    # Combine semantic labels for things and stuff
    sem_seg_result = torch.zeros((H, W), dtype=torch.long, device=device)  # Create tensor on the same device
    sem_seg_result[thing_mask] = thing_semantic[thing_mask]
    sem_seg_result[stuff_mask] = stuff_semantic[stuff_mask]
    sem_seg_result = sem_seg_result.unsqueeze(0)
    return sem_seg_result


def merge_semantic_and_instance(
        sem_seg, ins_seg, semantic_thing_seg, label_divisor, thing_ids, stuff_area, void_label, DBSCAN_run, outliers_mask
):
    # In case thing mask does not align with semantic prediction.
    pan_seg = torch.zeros_like(sem_seg) + void_label
    is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)
    # Keep track of instance id for each class.
    class_id_tracker = Counter()
    # Paste thing by majority voting.
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within `semantic_thing_seg`.
        thing_mask = (ins_seg == ins_id) & is_thing
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
        class_id_tracker[class_id.item()] += 1
        new_ins_id = class_id_tracker[class_id.item()]
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id
    # Paste stuff to unoccupied area.
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_ids:
            # thing class
            continue
        # Calculate stuff area.
        if DBSCAN_run:
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0) | outliers_mask
        else:
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        if stuff_mask.sum().item() >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_panoptic_segmentation(
        total_center,
        sem_logit,
        embedding,
        uncertainty,
        association_score,
        thing_ids,
        label_divisor,
        stuff_area,
        void_label,
        foreground_mask=None,
):
    """
    :param total_center: An integer, the number of detected centers.
    :param sem_logit: A Tensor of shape [1, H, W]
    :param embedding: A Tensor of shape [8, H, W] of raw embedding output.
    :param uncertainty: A Tensor of shape [1, H, W] of raw uncertainty output.
    :param association_score: A Tensor of shape [211, H, W] of raw association score output.
    :param thing_ids: A set of ids from contiguous category ids belonging
            to thing categories.
    :param label_divisor: An integer, used to convert panoptic id =
            semantic id * label_divisor + instance_id.
    :param stuff_area: An integer, remove stuff whose area is less tan stuff_area.
    :param void_label: An integer, indicates the region has no confident prediction.
    :param foreground_mask:
    :return:
    """
    valid_thing_association_score = association_score[:total_center, :, :]
    valid_stuff_association_score = association_score[total_center:total_center+11, :, :]
    valid_association_score = association_score[:total_center+11, :, :]

    pixel_classes = torch.argmax(valid_association_score, dim=0)
    thing_seg = (pixel_classes < total_center)

    sem_seg = get_semantic_segmentation(
        sem_logit,
        thing_seg,
        valid_stuff_association_score,
    )

    ins_seg, DBSCAN_run, outliers_mask = get_instance_segmentation(
        sem_seg,
        total_center,
        valid_thing_association_score,
        thing_seg,
        embedding,
        uncertainty,
    )

    panoptic = merge_semantic_and_instance(
        sem_seg, ins_seg, thing_seg, label_divisor, thing_ids, stuff_area, void_label, DBSCAN_run, outliers_mask
    )
    return panoptic, sem_seg
