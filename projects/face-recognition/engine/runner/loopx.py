import queue as Queue
import threading

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.runner import EpochBasedTrainLoop

from mmrazor.registry import LOOPS


@LOOPS.register_module()
class EpochBasedTrainLoopX(EpochBasedTrainLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin=val_begin,
                         val_interval=val_interval,
                         dynamic_intervals=dynamic_intervals)
        # Make a dataloader proxy.
        # import pdb; pdb.set_trace()
        self.dataloader = DataLoaderX(self.dataloader)


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX:

    def __init__(self, dataloader: DataLoader):
        self.local_rank = torch.cuda.current_device()
        self.stream = torch.cuda.Stream(self.local_rank)
        self._dataloader = dataloader

    def __iter__(self):
        self.iter = self._dataloader.__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        assert 'inputs' in self.batch        
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch['inputs'])):
                self.batch['inputs'][k] = self.batch['inputs'][k].to(
                    device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __len__(self):
        return len(self._dataloader)

    @property
    def dataset(self):
        return self._dataloader.dataset
