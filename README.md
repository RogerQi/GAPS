# GAPS

**Official implementation for CVPRW2023 Paper: GAPS: Few-Shot Incremental Semantic Segmentation via Guided Copy-Paste Synthesis**

*Ri-Zhao Qiu, Peiyi Chen, Wangzhe Sun, Yu-Xiong Wang, and Kris Hauser*

**[[Paper](https://github.com/IssamLaradji/l3divu/blob/main/l3divu_2023/accepted_no_archive/10.pdf)]**
**[[Poster](https://drive.google.com/file/d/1BmaOtSFQjQgi96xOAQDocgQ27Jy46Qvd/view?usp=sharing)]**

# Preparation

## Setup dependencies

```bash
conda create --name GAPS python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 cudatoolkit-dev=11.6 -c pytorch -c conda-forge -c nvidia
pip install -r requirements.txt
```

## Prepare ImageNet-Pretrained Models

Like many other few-shot/incremental/general segmentation works, GAPS is trained from ImageNet pretrained weights.

```bash
mkdir pretrained_model
cd pretrained_model
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
```

## Prepare Dataset

By default, the dataset root is `./data`. Alternatively, you can specify your own dataset root by either setting an environmental variable `$DATASET_ROOT` or linking the data folder. For more details you can refer to https://github.com/RogerQi/dl_codebase/blob/roger/submission/modules/utils/misc.py#L10.

Let's take `./data` as an example. To start with, create the data folder.

```bash
mkdir data
```

### Pascal-5<sup>i</sup>

Pascal segmentation datasets usually contain two sets of datasets - the original segmentation mask accompanying Pascal VOC 2012 semantic segmentation challenge, and a set of additional annotations supplemented by Berkeley SBD project.

Fortunately, torchvision has routines for conveniently downloading both of these two sets. The codebase contains code for automatically downloading these two datasets. You can run,

```bash
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split0_5shot.yaml
```

and the Pascal-5i dataset will automatically be ready. You can expect to see the training process begin. Hit `Ctrl+C` to interrupt it.

### COCO-20i<sup>i</sup>

To use the COCO dataset, you need to manually obtain it.

```bash
cd data
mkdir COCO2017
# COCO2017 training images
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
# val images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
# Stuff-Things semantic annotations map
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
unzip stuffthingmaps_trainval2017.zip
```

To test if the COCO dataset is working, you can run

```bash
python3 main/train.py --cfg configs/fs_incremental/coco20i_split0_5shot.yaml
```

# Running GAPS

As described in our paper, learning in GAPS are divided into two stages: base learning stage and incremental learning stage.

## Base learning stage

```bash
# run from project root
cd GAPS
# Base training on Pascal-5i (note that 5-shot and 1-shot share same base weights)
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split0_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split1_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split2_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml
# Base training on COCO-20i
python3 main/train.py --cfg configs/fs_incremental/coco20i_split0_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/coco20i_split1_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/coco20i_split2_5shot.yaml
python3 main/train.py --cfg configs/fs_incremental/coco20i_split3_5shot.yaml
```

Empirically, the entire base learning stage takes approximately 5 days on a machine with a single RTX 3090 GPU.

If you want to skip base learning, you can find weights trained from the base stage in the table below.

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

## Incremental learning stage

Take the split-3 of the Pascal-5i dataset as an example.
After the base learning stage, the codebase will generate a weight named `GIFS_pascal_voc_split3_final.pt` at the project root. To perform incremental learning and testing, the command line to be invoked is

```bash
python3 main/test.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml --load GIFS_pascal_voc_split3_final.pt
```

and you should see the results. Note that the diversity-guided exemplar selection requires computation of prototype of every image in the base training stage, which requires roughly 15 minutes on the first time one runs incremental learning on a split.
