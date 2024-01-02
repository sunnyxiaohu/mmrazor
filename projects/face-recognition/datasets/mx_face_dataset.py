import numbers
import os
import random

from typing import Callable, List, Sequence, Union

import mxnet as mx
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend
from mmengine.logging import print_log

from mmrazor.registry import DATASETS


@DATASETS.register_module()
class MXFaceDataset(BaseDataset):

    def __init__(self, data_root, pipeline: List[Union[dict, Callable]] = []):
        path_imgrec = os.path.join(data_root, 'train.rec')
        path_imgidx = os.path.join(data_root, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec,
                                                    'r')
        super(MXFaceDataset, self).__init__(
            data_root=data_root, serialize_data=False, pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            data_list = list(range(1, int(header.label[0])))
        else:
            data_list = list(self.imgrec.keys)
        return data_list

    def get_data_info(self, idx: int) -> dict:
        idx = self.data_list[idx]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        sample = mx.image.imdecode(img).asnumpy()
        sample = sample[..., ::-1]  # rgb to bgr
        info = dict(img=sample, gt_label=int(label))
        return info


@DATASETS.register_module()
class MatchFaceDataset(BaseDataset):

    def __init__(self,
                 dataset_name: str,
                 key_file: str,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 sample_ratio: float = None,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 pipeline: List[Union[dict, Callable]] = []):

        self.dataset_name = dataset_name
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.sample_ratio = sample_ratio
        super(MatchFaceDataset, self).__init__(
            ann_file=key_file,
            data_root=data_root,
            data_prefix=data_prefix,
            serialize_data=False,
            pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        # Note that there may be images missing in the image forler
        # or unused in the key_file,
        # so, we get samples from image folder and filter the missing
        # samples or unused samples from the key_file for double checking.
        data_list = []
        all_samples1 = []
        self.sample_idx_mapping = {}
        backend = get_file_backend(self.data_root, enable_singleton=True)
        files = backend.list_dir_or_file(
            self.data_prefix['img_path'],
            list_dir=False,
            list_file=True,
            recursive=False,
        )
        for file in sorted(list(files)):
            if self.is_valid_file(file):
                path = backend.join_path(self.data_prefix['img_path'], file)
                data_list.append(path)
                all_samples1.append(file.split('.')[0])

        key_file = os.path.join(self.data_root, self.ann_file)
        with open(key_file, 'r') as f:
            mappings = list(f.readlines())
        total_images = int(mappings.pop(0))
        assert len(mappings) == total_images
        all_samples2 = [data.strip().split(' ')[0] for data in mappings]

        all_samples = set(all_samples1) & set(all_samples2)
        if len(all_samples) != len(all_samples1) or len(all_samples) != len(
                all_samples2):
            print_log(
                f'Samples in "{self.dataset_name}" may lost, ' +
                f'in folder: {len(all_samples1)}, in key_file: {len(all_samples2)}. '
                + f'Finally, we adjust to {len(all_samples)}',
                logger='current')
        # filter and select samples
        data_list = list(
            filter(
                lambda path: path.split('/')[-1].split('.')[0] in all_samples,
                data_list))
        if self.sample_ratio is not None:
            assert self.sample_ratio > 0  and self.sample_ratio <= 1
            org_len = len(data_list)
            new_len = int(self.sample_ratio * org_len)
            data_list = random.sample(data_list, new_len)
            print_log(
                f'Sample "{new_len}" from total {org_len} data_list in "{self.dataset_name}"',
                logger='current')

        # align dix with data_list
        all_samples = [path.split('/')[-1].split('.')[0] for path in data_list]

        self.sample_idx_mapping = dict(
            zip(all_samples, range(len(all_samples))))
        self.sample_label_mapping = {}
        self.sample_idx_identical_mapping = {}
        num_classes = -1
        for data in mappings:
            # current_img, identical_nums, identical_img1, identical_img2, ...
            tockens = data.strip().split(' ')
            if tockens[0] not in all_samples:
                continue
            # set the current_img and its corresponding identical images with gt_label
            if tockens[0] not in self.sample_label_mapping:
                num_classes += 1
                self.sample_label_mapping[tockens[0]] = num_classes
            identical_idxs = []
            for identical_img in tockens[2:]:
                if identical_img not in all_samples:
                    continue
                if identical_img not in self.sample_label_mapping:
                    self.sample_label_mapping[identical_img] = num_classes
                identical_idxs.append(self.sample_idx_mapping[identical_img])

            self.sample_idx_identical_mapping[self.sample_idx_mapping[
                tockens[0]]] = identical_idxs

        # self._metainfo['num_classes'] = num_classes
        # self._metainfo['sample_idx_mapping'] = self.sample_idx_mapping
        # self._metainfo['sample_label_mapping'] = self.sample_label_mapping
        # self._metainfo['sample_idx_identical_mapping'] = self.sample_idx_identical_mapping
        return data_list

    def get_data_info(self, idx: int) -> dict:
        img_path = self.data_list[idx]
        sample = img_path.split('/')[-1].split('.')[0]
        info = {
            'img_path': img_path,
            'gt_label': self.sample_label_mapping[sample],
            'identical_idx': self.sample_idx_identical_mapping[idx],
            'sample_idx': idx,
            # Used for sample_identical_mapping
            'sample_idx_identical_mapping': self.sample_idx_identical_mapping,
            # Compatible with multiple dataset (ConcatDataset).
            'dataset_name': self.dataset_name,
        }
        return info

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
