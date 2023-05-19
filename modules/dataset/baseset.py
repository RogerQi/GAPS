import numpy as np
import torch
from torchvision import transforms
from .transforms.dispatcher import dispatcher

class JointCompose:
    '''
    Resembles

    https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose

    but it works with joint transformation (i.e., both images and label map)
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return (img, target)

class base_set(torch.utils.data.Dataset):
    '''
    An implementation of torch.utils.data.Dataset that supports various
    data transforms and augmentation.
    '''
    def __init__(self, dataset, split, cfg):
        '''
        Args:
            dataset: any object with __getitem__ and __len__ methods implemented.
                Object retruned from dataset[i] is expected to be (raw_tensor, label).
            split: ("train" or "test"). Specify dataset mode
            cfg: yacs root config node object.
        '''
        assert split in ["train", "test"]
        self.cfg = cfg
        self.dataset = dataset
        self.split = split

        train_ops, joint_train_ops = dispatcher(cfg.DATASET.TRANSFORM.TRAIN)
        self.train_data_transforms = self._get_mono_transforms(cfg.DATASET.TRANSFORM.TRAIN, train_ops)
        self.train_joint_transforms = self._get_joint_transforms(cfg.DATASET.TRANSFORM.TRAIN, joint_train_ops)

        test_ops, joint_test_ops = dispatcher(cfg.DATASET.TRANSFORM.TEST)
        self.test_data_transforms = self._get_mono_transforms(cfg.DATASET.TRANSFORM.TEST, test_ops)
        self.test_joint_transforms = self._get_joint_transforms(cfg.DATASET.TRANSFORM.TEST, joint_test_ops)
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            assert isinstance(key[0], int) or isinstance(key[0], np.integer)
            assert isinstance(key[1], dict)
            index = key[0]
            params = key[1]
        elif isinstance(key, int) or isinstance(key, np.integer):
            index = key
            params = {}
        else:
            raise NotImplementedError
        data, label = self.dataset[index]

        # Flag which determines if train transform is used or test transforms
        aug_flag = (self.split == 'train')
        # Augmentation flag can be manually overriden
        if 'aug' in params:
            assert params['aug'] in [True, False]
            aug_flag = params['aug']

        if aug_flag:
            data = self.train_data_transforms(data)
            data, label = self.train_joint_transforms(data, label)
        else:
            data = self.test_data_transforms(data)
            data, label = self.test_joint_transforms(data, label)

        return (data, label)

    def __len__(self):
        return len(self.dataset)

    def _get_mono_transforms(self, transforms_cfg, transform_ops_list):
        transforms_list = transforms_cfg.transforms
        assert len(transforms_list) != 0
        if transforms_list == ('none',):
            return transforms.Compose([])
        # Nontrivial transforms...
        try:
            normalize_first_occurence = transforms_list.index("normalize")
            assert normalize_first_occurence == len(transforms_list) - 1, "normalization happens last"
            return transforms.Compose(transform_ops_list + [transforms.ToTensor(),
                        self._get_dataset_normalizer(transforms_cfg)])
        except ValueError:
            # Given transforms does not contain normalization
            return transforms.Compose(transform_ops_list + [transforms.ToTensor()])
    
    def _get_joint_transforms(self, transforms_cfg, transforms_ops_list):
        return JointCompose(transforms_ops_list)

    def _get_dataset_normalizer(self, transforms_cfg):
        return transforms.Normalize(mean=transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    std=transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.sd)
