def dataset_dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    if dataset_name == "mnist":
        import dataset.mnist
        return dataset.mnist
    elif dataset_name == "cifar10":
        import dataset.cifar10
        return dataset.cifar10
    elif dataset_name == "imagenet":
        raise NotImplementedError
    elif dataset_name == "mini_imagenet":
        import dataset.mini_imagenet
        return dataset.mini_imagenet
    elif dataset_name == "mini_imagenet_w_bg":
        import dataset.mini_imagenet_w_bg
        return dataset.mini_imagenet_w_bg
    elif dataset_name == "numpy":
        import dataset.generic_np_dataset
        return dataset.generic_np_dataset
    elif dataset_name == "coco2017":
        import dataset.coco
        return dataset.coco
    elif dataset_name == "ade20k":
        import dataset.ade20k
        return dataset.ade20k
    elif dataset_name == "VOC2012_seg":
        import dataset.voc2012_seg
        return dataset.voc2012_seg
    elif dataset_name == "pascal_5i":
        import dataset.pascal_5i
        return dataset.pascal_5i
    elif dataset_name == "voc2012_5i":
        import dataset.voc2012_5i
        return dataset.voc2012_5i
    elif dataset_name == "scannet_25k":
        import dataset.scannet_25k
        return dataset.scannet_25k
    elif dataset_name == "coco_20i":
        import dataset.coco_20i
        return dataset.coco_20i
    elif dataset_name == "places365_stanford":
        import dataset.places365_stanford
        return dataset.places365_stanford
    elif dataset_name == "ade20k_incremental":
        import dataset.ade20k_incremental
        return dataset.ade20k_incremental
    else:
        raise NotImplementedError
