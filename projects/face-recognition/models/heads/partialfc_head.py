from typing import List, Optional, Tuple, Union

import torch
from torch import distributed
import torch.nn as nn
from torch.nn.functional import linear, normalize

from mmrazor.registry import MODELS

try:
    from mmcls.structures import ClsDataSample
    from mmcls.models.heads.base_head import BaseHead
except ImportError:
    from mmrazor.utils import get_placeholder
    ClsDataSample = get_placeholder('mmcls')
    BaseHead = get_placeholder('mmcls')


@MODELS.register_module()
class PartialFCHead(BaseHead):
    """Classification head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 embedding_size: int,
                 num_classes: int,
                 sample_rate: float = 1.0,
                 fp16: bool = False,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),                 
                 init_cfg: Optional[dict] = None):
        super(PartialFCHead, self).__init__(init_cfg=init_cfg)
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_mom: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_mom: torch.Tensor
        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_mom",
                tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_mom",
                tensor=torch.empty(0, 0))
            self.register_buffer("weight_index",
                tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        # self.step = 0

    @torch.no_grad()
    def sample(self, labels, index_positive):
        # self.step += 1
        positive = torch.unique(labels[index_positive], sorted=True).cuda()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).cuda()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].cuda()
            index = index.sort()[0].cuda()
        else:
            index = positive
        self.weight_index = index

        labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_mom = self.weight_mom[self.weight_index]

        # if isinstance(optimizer, torch.optim.SGD):
        #     # TODO the params of partial fc must be last in the params list
        #     optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
        #     optimizer.param_groups[-1]["params"][0] = self.weight_activated
        #     optimizer.state[self.weight_activated][
        #         "momentum_buffer"
        #     ] = self.weight_activated_mom
        # else:
        #     raise

    @torch.no_grad()
    def update(self):
        """ partial weight to global
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_mom[self.weight_index] = self.weight_activated_mom

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``PartialFCHead``, we just obtain the feature
        of the last stage.
        """
        # The PartialFCHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The PartialFCHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.cat([i.gt_label.label for i in data_samples])

        target.squeeze_()
        target = target.long()
        self.update()

        batch_size = cls_score.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
            self.last_batch_size, batch_size))

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(cls_score, *_gather_embeddings)
        distributed.all_gather(_gather_labels, target)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        # compute loss
        losses = dict()
        loss = self.loss_module(logits, labels)
        losses['loss'] = loss
        optimizer = dict()
        optimizer['params'] = self.weight_activated
        optimizer['state'] = dict(momentum_buffer=self.weight_activated_mom)
        losses['optimizer'] = optimizer

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[Union[ClsDataSample, None]] = None
    ) -> List[ClsDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ClsDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    # TODO(shiguang): check state_dict and load_state_dict.

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = cls_score
        # pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if data_sample is None:
                data_sample = ClsDataSample()

            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
