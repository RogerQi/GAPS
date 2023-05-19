# GAPS

**Official implementation for CVPRW2023 Paper: GAPS: Few-Shot Incremental Semantic Segmentation via Guided Copy-Paste Synthesis**

*Ri-Zhao Qiu, Peiyi Chen, Wangzhe Sun, Yu-Xiong Wang, and Kris Hauser*

**[[Paper](http://motion.cs.illinois.edu/papers/CVPRWorkshop2023-Qiu-FewShotSegmentation.pdf)]**

*Code and instructions have not been fully organized and tested. I will get back to this after the NeurIPS supplementary material deadline.*

# Prepare Dataset

Before preparing dataset, you first need to determine dataset root. You can set dataset root by setting a system-wide environment variable $DATASET_ROOT. If the environmental variable is not set, by default, it uses `/data`. For more details you can refer to https://github.com/RogerQi/dl_codebase/blob/roger/submission/modules/utils/misc.py#L10.

In the following instructions, we will assume that the data root is `/data`.

Main experiments of GAPS are done on two datasets: pascal-5<sup>i</sup> and coco-20<sup>i</sup>.

## Pascal-5<sup>i</sup>

Pascal segmentation datasets usually contain two sets of datasets - the original segmentation mask accompanying Pascal VOC 2012 semantic segmentation challenge, and a set of additional annotations supplemented by Berkeley SBD project.

Fortunately, torchvision has routines for conveniently downloading both of these two sets. The easiest way to download these two datasets are running examples at https://github.com/RogerQi/pascal-5i/blob/main/examples.ipynb

## COCO-20i<sup>i</sup>

TBD

## ImageNet-Pretrained Models

Like many other few-shot/incremental/general segmentation works, GAPS is trained from ImageNet pretrained weights. In particular, for fair comparisons with existing works, we follow their implementations and also use ResNet-101. The pretrained weights can be downloaded from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth [Other ResNet weights can be found here](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html).

By default, the model loads weights from `/data/pretrained_model/resnet101-5d3b4d8f.pth` as defined [here](https://github.com/RogerQi/dl_codebase/blob/roger/submission/configs/fs_incremental/pascal5i_base.yaml#L16).

## Running GAPS

As described in our paper, learning in GAPS are divided into two stages: base learning stage and incremental learning stage. Take Pascal-5-3 as an example. To run the base learning stage, the command line to invoke is

```
cd dl_codebase # run from project root
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml
```

If you want to skip the prolonged base learning stage, you can find weights trained from the base stage in the table below.

| Dataset | Base IoU | Novel IoU | Base weights |
| --- | --- | --- | --- |
| Pascal-5-0 | TBD | TBD | [box](https://uofi.box.com/s/qwjpio1xubzp2h87vzmnosvff3kt2sfz) |
| Pascal-5-1 | TBD | TBD | [box](https://uofi.box.com/s/3b4opya1qmhztnn2mxaqjce32izvuvep) |
| Pascal-5-2 | TBD | TBD | [box](https://uofi.box.com/s/s9tb3jcl2n1vi73iu2e482to1txfhyhs) |
| Pascal-5-3 | TBD | TBD | [box](https://uofi.box.com/s/1fhmkne8pm8l8ucsg4uazlisioito8f1) |
| COCO-20-0  | TBD | TBD | [box](https://uofi.box.com/s/wnk7rscz9py9hkufqr78d130s8o3mgtx) |
| COCO-20-1  | TBD | TBD | [box](https://uofi.box.com/s/ccrosqwpks20ik5u50btxyfhf5776mjn) |
| COCO-20-2  | TBD | TBD | [box](https://uofi.box.com/s/83fgz1jxjrgxyb1i6ff4f8oowubr74vb) |
| COCO-20-3  | TBD | TBD | [box](https://uofi.box.com/s/y8hlhvzhseomtj6fjutyylt68v8jl1e6) |

After base learning stage, it will generate a weight named `GIFS_pascal_voc_split3_final.pt` at the project root. To perform incremental learning and testing, the command line to be invoked is

```
python3 main/test.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml --load GIFS_pascal_voc_split3_final.py
```

and you should see the results.
