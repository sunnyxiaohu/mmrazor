##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from typing import Callable, Dict, List, Optional

import os, sys, hashlib
import numpy as np
from PIL import Image
import pickle

from mmengine.fileio import exists, get, join_path
from mmrazor.registry import DATASETS

try:
    from mmcls.datasets.base_dataset import BaseDataset
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDataset = get_placeholder('mmcls')


@DATASETS.register_module()
class ImageNet16(BaseDataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf
    url_prefix = 'https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4?usp=sharing'

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self,
                 data_prefix: str,
                 test_mode: bool,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 download: bool = True,
                 class_nums = None,
                 **kwargs):
        self.download = download
        self.class_nums = class_nums
        super().__init__(
            # The MNIST dataset doesn't need specify annotation file
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        root = self.data_prefix['root']

        assert self._check_exists(), \
            'Download failed or shared storage is unavailable. Please ' \
            f'download the dataset manually through {self.url_prefix}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list

        imgs = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = join_path(root, file_name)
            entry = pickle.loads(get(file_path), encoding='latin1')
            imgs.append(entry['data'])
            if 'labels' in entry:
                gt_labels.extend(entry['labels'])
            else:
                gt_labels.extend(entry['fine_labels'])

        imgs = np.vstack(imgs).reshape(-1, 3, 16, 16)
        imgs = imgs.transpose((0, 2, 3, 1))  # convert to HWC

        data_list = []
        for img, gt_label in zip(imgs, gt_labels):
            if self.class_nums is not None and gt_label > self.class_nums:
                continue
            info = {'img': img, 'gt_label': int(gt_label) - 1}
            data_list.append(info)
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [f"Prefix of data: \t{self.data_prefix['root']}"]
        return body

    def _check_exists(self):
        """Check the exists of data files."""
        root = self.data_prefix['root']

        for filename, _ in (self.train_list + self.valid_list):
            # get extracted filename of data
            fpath = join_path(root, filename)
            if not exists(fpath):
                return False
        return True
