# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional, Tuple, Union
from functools import partial
import torch.nn
import copy
from mmrazor.registry import TASK_UTILS
from mmrazor.models.task_modules.estimators.base_estimator import BaseEstimator
import onnx
from .rmadd0 import *
import npuc
import time
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from mmengine.analysis import get_model_complexity_info


def collate_fn_coco(batch):
    return tuple(zip(*batch))

@TASK_UTILS.register_module()
class OVEstimator(BaseEstimator):
    def __init__(
        self,
        task_type: str = 'det' ,
        input_shape: Tuple = (1, 3, 384, 640),
        units: Dict = dict(ov_ddr_bandwidth='M', ov_npu_time='ms'),
        as_strings: bool = False,
        ov_file_path: str = 'root_path',
        input_img_prefix: str = 'tmp.jpg',
        onnx_file_prefix: str = 'tmp.onnx',
        qfnodes_def_file_prefix: str = 'qfnode',
        val_root_prefix: str = 'val_root',
        val_annFile_prefix: str = 'val_annFile',
        ovm_file_preifx: str = 'temp.ovm',
        
    ):
        super().__init__(input_shape, units, as_strings)
        if not isinstance(units, dict):
            raise TypeError('units for estimator should be a dict',
                            f'but got `{type(units)}`')
        self.task_type = task_type
        self.ov_file_path = ov_file_path
        self.input_img_path = os.path.join(self.ov_file_path,input_img_prefix)
        self.onnx_file = os.path.join(self.ov_file_path,onnx_file_prefix)
        self.qfnodes_def_file =os.path.join(self.ov_file_path, qfnodes_def_file_prefix)
        self.val_root = os.path.join(self.ov_file_path,val_root_prefix)
        self.val_annFile =os.path.join(self.ov_file_path,val_annFile_prefix)
        self.ovm_file = os.path.join(self.ov_file_path,ovm_file_preifx)
            
    def pre_func(self,index):
        '''Pre-processsing function; for loading input images.'''
        print('\r{0}'.format(index), end='')
        # Get the images and labels; this is returned per transform()
        self.data, self.label = next(self.it_val_data, (None, None))
        x= self.data[0].numpy()    
        return x
    
    def reset_data(self):
        '''Reset dataset iterator.'''
        self.it_val_data = iter(self.val_data)
        
    def estimate(self,
                 model: torch.nn.Module,
                 ) -> Dict[str, Union[float, str]]:
        
        resource_metrics = dict()
        
        model.eval()
        cpu_model = copy.deepcopy(model.cpu())
        analysis_results = get_model_complexity_info(
            cpu_model,
            (self.input_shape[1],self.input_shape[2],self.input_shape[3])
        )
        normalize_cfg=None
        # to onnx
        if self.task_type=='cls':
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
            img = img.astype(np.float)
            img = (img / 255. - 0.5) / 0.5  # torch style norm
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            torch.onnx.export(
                cpu_model, 
                img, 
                self.onnx_file, #output, 
                keep_initializers_as_inputs=False, 
                verbose=False, 
                opset_version=11)
        elif self.task_type=='det':
            from .pytorch2onnx_det import pytorch2onnx
            pytorch2onnx(
                cpu_model,
                self.input_img_path,
                self.input_shape,
                normalize_cfg,
                opset_version=11,
                show=False,
                output_file=self.onnx_file,
                verify=False,
                test_img=None,
                do_simplify=False,
                dynamic_export=None,
                skip_postprocess=True)
        onnx_model = onnx.load(self.onnx_file)
        graph = onnx_model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)
        onnx.save(onnx_model, self.onnx_file)
        onnx_file = self.onnx_file
        #  to ovm
        try:
            graph,params = npuc.frontend.from_onnx(onnx_file)
            print ('onnx convert finished')
            graph,params = npuc.graphopt(graph,params)
            print ('graph optimize finished')
            ifm = 'data'
            ov_shape = { ifm : self.input_shape }
            ovm_hdl = npuc.get_ovm_hdl(graph,params,ov_shape,qfnodes_fn=self.qfnodes_def_file)
            data_transform = {
                "val": transforms.Compose([transforms.Resize(self.input_shape[-1]),
                                        transforms.CenterCrop((self.input_shape[-2],self.input_shape[-1])),
                                        transforms.ToTensor()])} #,
            coco_val = datasets.CocoDetection(self.val_root,self.val_annFile,transform=data_transform["val"])
            self.val_data = torch.utils.data.DataLoader(coco_val, batch_size=1 ,shuffle=False,num_workers=1,pin_memory=True,collate_fn=collate_fn_coco,drop_last=False)
            self.reset_data()
            t1 = time.time()
            ovm_hdl = npuc.calib.run(ovm_hdl, len(self.val_data), {'func':self.pre_func})
            self.reset_data()
            ovm_hdl = npuc.resolve_q(ovm_hdl)
            t2 = time.time()
            print ('calibrate consume: ',str(t2-t1),' s')
            print ('calibrate finished')
            t3 = time.time()
            npuc.ovm.gen(self.ovm_file,ovm_hdl,compression=False)
            t4 = time.time()
            print ('ovm gen consume: ',str(t4-t3),' s')
            ppa = npuc.ovm.ppa(self.ovm_file, 0, self.input_shape[-1], self.input_shape[-2], 3, 8, False)
            # print('PPA information:')
            # print('NPU processing time     = {0:8.3f} S    for 1 inference'.format(ppa.time_frame))
            # print('NPU total ops           = {0:8.3f} GOPs for 1 inference'.format(ppa.ops_frame))
            # print('NPU total ddr bandwidth = {0:8.3f} MB   for 1 inference'.format(ppa.bandwidth_frame))
            # print('NPU average power       = {0:8.3f} mW    at 1 inference per second (indicative only)'.format(ppa.power_frame))
            resource_metrics.update({
            'ov_ddr_bandwidth': ppa.bandwidth_frame,
            'ov_npu_time': ppa.time_frame,
            'ov_params': float(analysis_results['params_str'][:-1])
            })
            
        except:
            print ('compiler error!!!')
            resource_metrics.update({
                'ov_ddr_bandwidth': 1000,
                'ov_npu_time': 1000,
                'ov_params': 1000
            })
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        print (resource_metrics)
        return resource_metrics

    