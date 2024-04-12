# MaskFormer Model Zoo and Baselines

## Introduction

This file documents a collection of models reported in our paper.
All numbers were obtained on 2 NVIDIA A40 GPUS

#### How to Read the Tables
* The "Name" column contains a link to the config file. Running `train_net.py --num-gpus 2` with this config file
  will reproduce the model.
* Training curves and other statistics can be found in `metrics` for each model.

### Holistic Segmentation Models

#### Cityscapes holistic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">PQ (open-cityscapes)</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: u3hs_R_52_90k_bs16_crop_512_1024 -->
 <tr><td align="left"><a href="../configs/Cityscapes-PanopticSegmentation/u3hs_R_52_90k_bs16_crop_512_1024.yaml.yaml">U3HS-reimplemented</a></td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">40.671</td>
<td align="center"><a href="https://drive.google.com/file/d/1-4gyFUEK3xHzy_-F_q-jT7xahwX8yl-L/view?usp=share_link">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1-CeG1Sl56EEzcPFoE95hgRExSgNemAqb/view?usp=sharing">metrics</a></td>
</tbody></table>
