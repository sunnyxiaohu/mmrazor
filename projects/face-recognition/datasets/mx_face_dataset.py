import numbers
import os
import mxnet as mx
from typing import Callable, List, Sequence, Union

from mmengine.fileio import get_file_backend
from mmengine.dataset import BaseDataset
from mmrazor.registry import DATASETS


@DATASETS.register_module()
class MXFaceDataset(BaseDataset):
    def __init__(self, data_root, pipeline: List[Union[dict, Callable]] = []):
        path_imgrec = os.path.join(data_root, 'train.rec')
        path_imgidx = os.path.join(data_root, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec, 'r')
        super(MXFaceDataset, self).__init__(data_root=data_root,
                                            serialize_data=False,
                                            pipeline=pipeline)

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
        info = dict(img=sample, gt_label=int(label))
        return info


@DATASETS.register_module()
class MatchFaceDataset(BaseDataset):
    def __init__(self, key_file: str, data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                    '.bmp', '.pgm', '.tif'),
                 pipeline: List[Union[dict, Callable]] = []):

        self.extensions = tuple(set([i.lower() for i in extensions]))
        super(MatchFaceDataset, self).__init__(ann_file=key_file,
                                               data_root=data_root,
                                               data_prefix=data_prefix,
                                               serialize_data=False,
                                               pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        # Note that there may be images missing in the image forler,
        # so, we get samples from image folder and filter the missing
        # samples from the key_file.
        samples = []
        self.sample_idx_mapping = {}
        backend = get_file_backend(self.data_root, enable_singleton=True)
        files = backend.list_dir_or_file(
            self.data_prefix['img_path'],
            list_dir=False,
            list_file=True,
            recursive=False,
        )
        idx = 0
        for file in sorted(list(files)):
            if self.is_valid_file(file):
                path = backend.join_path(self.data_prefix['img_path'], file)
                samples.append(path)
                self.sample_idx_mapping[file.split('.')[0]] = idx
                idx += 1

        key_file = os.path.join(self.data_root, self.ann_file)
        all_samples = list(self.sample_idx_mapping)
        with open(key_file, 'r') as f:
            mappings = list(f.readlines())
        total_images = int(mappings.pop(0))
        assert len(mappings) == total_images
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

            self.sample_idx_identical_mapping[self.sample_idx_mapping[tockens[0]]] = identical_idxs

        self._metainfo['num_classes'] = num_classes
        self._metainfo['sample_idx_mapping'] = self.sample_idx_mapping
        self._metainfo['sample_label_mapping'] = self.sample_label_mapping
        self._metainfo['sample_idx_identical_mapping'] = self.sample_idx_identical_mapping
        return samples

    def get_data_info(self, idx: int) -> dict:
        img_path = self.data_list[idx]
        sample = img_path.split('/')[-1].split('.')[0]
        info = {
            'img_path': img_path,
            'gt_label': self.sample_label_mapping[sample],
            'identical_idx': self.sample_idx_identical_mapping[idx],
            'sample_idx': idx
        }
        return info

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
