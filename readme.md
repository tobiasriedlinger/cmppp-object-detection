# Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection

Object detection models based on conditional marked Poisson point processes (CMPPP) as proposed in "Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection".
Due to the requirements of the model, this repository contains the evaluation code. Model implementation can be found in forked repositories of MMDetection (https://github.com/tobiasriedlinger/cmppp-object-detection-mmdetection) and MMSegmentation (https://github.com/tobiasriedlinger/cmppp-object-detection-mmsegmentation), respectively.

## Environment Setup
The implementation was used under Python 3.8.18, CUDA 12 with MMCV2.1.0, MMDetection 3.3.0 and MMSegmentation 1.2.2. 
Particularly, the following repositories contain the implementations of the point process models:
- https://github.com/tobiasriedlinger/cmppp-object-detection-mmdetection
- https://github.com/tobiasriedlinger/cmppp-object-detection-mmsegmentation

Other specifications concerning package versions can be found in `requirements.txt`.
The following files contain the majority of the implementation:
```
/mmdetection
    /configs/
        _base_/models/
            pppseg-dlv3-rn50_fpn.py 
            pppseg-hr48_fpn.py
        poisson_proto_net/
            pppseg-dlv3-rn50_fpn-cityscapes.py
            pppseg-fcn-hr48-cityscapes.py
    /mmdet/models/
        backbones/
            ppp_segmentor.py
        dense_heads/
            poisson_proto_head.py
        detectors/
            poisson_proto.py
    /tools/
        run.py
/mmsegmentation
    /configs/
        _base_/datasets/
            cityscapes_instances.py
        poisson/
            deeplabv3plus_r50-d8_4xb2-80k_cityscapes_inst-512x1024.py
            fcn_hr48_4xb2-160k_cityscapes_inst-512x1024.py
    /mmseg/
        datasets/
            cityscapes_instances.py
        models/losses/
            poisson_loss.py
    /tools/
        run.py
```

In addition, we altered the data augmentation on the Cityscapes dataset in `mmdetection/configs/_base_/datasets/cityscapes_detection.py` to
```python3
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize',
        scale=(1024, 512),
        keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', 
        scale=(1024, 512),
        keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

## Training
The Cityscapes dataset (or any other dataset) needs to be set up according to the [documentation of MMDetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/cityscapes). Training of either the non-marked PPP model or the marked PPP model (object detection) may then be done by calling e.g., 
```
python3 mmdetection/tools/train.py model_repo/mmdetection/configs/poisson_proto_net/pppseg-dlv3-rn50_fpn-cityscapes.py
```

## Inference
Generating outputs of the non-marked PPP model or the marked PPP model (object detection) may be done by calling e.g.,
```
python3 mmdetection/tools/run.py
```
or
```
python3 mmsegmentation/tools/run.py /mmsegmentation/configs/poisson/deeplabv3plus_r50-d8_4xb2-80k_cityscapes_inst-512x1024.py /work_dirs/ckpts/iter_96000.pth
```
after having set the respective paths to
- the config file
- the checkpoint file
- the input image directory
- the target directory

### Evaluation
Preparation: Edit all necessary paths stored in `evaluation/global_defs.py` and select the tasks to be executed by setting the corresponding boolean variable (`True`/`False`).
Run the evaluation code: 
```
python evaluate.py
```