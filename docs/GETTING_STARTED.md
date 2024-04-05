## Getting Started with U3HS

This document provides a brief intro of the usage of U3HS.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `ade20k-150/maskformer_R50_bs16_160k.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```
cd demo/
python demo.py --config-file ../configs/Cityscapes-PanopticSegmentation/u3hs_R_52_crop_512_1024_dsconv.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory, use `--output`.

## Training & Evaluation in Command Line

To train a model with 8 GPUs run:
```bash
python ./train_net.py --config-file configs/Cityscapes-PanopticSegmentation/u3hs_R_52_90k_bs16_crop_512_1024.yaml --num-gpus 2
```

Model evaluation can be done similarly:
```bash
python ./train_net.py --config-file configs/Cityscapes-PanopticSegmentation/u3hs_R_52_90k_bs16_crop_512_1024.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```
For more options, see `./train_net.py -h`.
