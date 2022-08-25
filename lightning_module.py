# from collections import OrderedDict
import numpy as np
import os
from pathlib import Path
import SimpleITK as sitk

import torch
import pytorch_lightning as pl

# from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from fix_model_state_dict import fix_model_state_dict
from get_kernels_strides import get_kernels_strides
from post_process import post_process


class LightningModule(pl.LightningModule):
    def __init__(self, cfg, transforms=None, post_transforms=None):
        super(LightningModule, self).__init__()
        self.cfg = cfg
#        self.lossfun = get_loss(cfg.Loss, train=True)
#        self.lossfun_valid = get_loss(cfg.Loss, train=False)
##        self.lossfun = DiceCELoss(to_onehot_y=True, softmax=True)
#        self.txt_logger = cfg.txt_logger
        self.transforms = transforms
        self.post_transforms = post_transforms

        if cfg.Model.arch == 'unetr':
            from monai.networks.nets import UNETR
            self.model = UNETR(**cfg.Model.params)
        elif cfg.Model.arch == 'dyn_unet':
            from monai.networks.nets import DynUNet
            kernels, strides = get_kernels_strides(
                cfg.Data.dataset.patch_size, cfg.Data.dataset.spacing)
            print('kernels:', kernels)
            print('strides:', strides)
            cfg.Model.params.kernel_size = kernels
            cfg.Model.params.strides = strides
            cfg.Model.params.upsample_kernel_size = strides[1:]
            
            if isinstance(cfg.Data.dataset.total_fold, int) and self.cfg.Data.dataset.total_fold > 1:
                self.model0 = DynUNet(**cfg.Model.params)
                self.model1 = DynUNet(**cfg.Model.params)
                self.model2 = DynUNet(**cfg.Model.params)
                self.model3 = DynUNet(**cfg.Model.params)
                self.model4 = DynUNet(**cfg.Model.params)
            else:                       
                self.model = DynUNet(**cfg.Model.params)
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')
        
        if cfg.Model.pretrained != 'None':
            if isinstance(self.cfg.Data.dataset.total_fold, int) and cfg.Data.dataset.total_fold > 1:
                for i in range(cfg.Data.dataset.total_fold):
                    # Load pretrained model weights
                    print(f'Loading: {cfg.Model.pretrained[i]}')
                    checkpoint = torch.load(cfg.Model.pretrained[i], map_location='cpu')
                    state_dict = checkpoint['state_dict']
                    if i == 0:
                        self.model0.load_state_dict(fix_model_state_dict(state_dict))
                    elif i == 1:
                        self.model1.load_state_dict(fix_model_state_dict(state_dict))
                    elif i == 2:
                        self.model2.load_state_dict(fix_model_state_dict(state_dict))
                    elif i == 3:
                        self.model3.load_state_dict(fix_model_state_dict(state_dict))
                    elif i == 4:
                        self.model4.load_state_dict(fix_model_state_dict(state_dict))
            else:
                # Load pretrained model weights
                print(f'Loading: {cfg.Model.pretrained}')
                checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
                state_dict = checkpoint['state_dict']
                self.model.load_state_dict(fix_model_state_dict(state_dict))
            
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg.Model.params.out_channels)
        self.post_label = AsDiscrete(to_onehot=cfg.Model.params.out_channels)

        self.valid_metrics_fun = dict()
        self.valid_metrics_fun['dice'] = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

    def forward(self, x):
        y = self.model(x)
        return y

    def forward_ensemble(self, x, idx):
        if idx == 0:
            y = self.model0(x)
        elif idx == 1:
            y = self.model1(x)
        elif idx == 2:
            y = self.model2(x)
        elif idx == 3:
            y = self.model3(x)
        elif idx == 4:
            y = self.model4(x)
        return y        
       
    def move_to(self, obj, device):
#        print('obj:', obj)
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        elif isinstance(obj, str):
            return obj
        else:
            print('obj (error):', obj)
            raise TypeError("Invalid type for move_to")

    def predict_step(self, batch, batch_idx):
        image = batch["CTres_SUV"]

        roi_size = self.cfg.Data.dataset.patch_size
        sw_batch_size = self.cfg.Data.dataset.sliding_window_batch_size

        if isinstance(self.cfg.Data.dataset.total_fold, int) and self.cfg.Data.dataset.total_fold > 1:
            assert image.shape[0] == 1, f"Batch size should be 1, but it is {image.shape[0]}."

            # prediction by each model
            preds = []
            for idx in range(self.cfg.Data.dataset.total_fold):
                pred = sliding_window_inference(
                    image, roi_size, sw_batch_size, self.forward_ensemble, idx=idx
                )
                preds.append(pred.cpu())
                
            # aggregate
            preds = torch.stack(preds, dim=0)
            preds = torch.mean(preds, dim=0)

            batch = self.move_to(batch, 'cpu')
            batch["pred"] = preds

            batch = [self.post_transforms(b) for b in decollate_batch(batch)]

            # save prediction results
            pred = batch[0]["pred"]                     # batch 0
            pred = pred[0].numpy().astype(np.uint8)     # channel 0
            
            if self.cfg.Model.post_process:
                # post processing (cc3d)
                pred = post_process(pred, num_classes=2)

            pred = pred.transpose(2, 1, 0)              # conver to (x,y,z) order for sitk
            pred = sitk.GetImageFromArray(pred)
            image_filename = batch[0]['CTres_meta_dict']['filename_or_obj']
            image = sitk.ReadImage(image_filename)
            pred.CopyInformation(image)
            output_filename = f'{self.cfg.Path.output_path}/PRED.nii.gz'
            print('Writing results with SITK format:', output_filename)
            sitk.WriteImage(pred, str(output_filename))

        else:
            batch["pred"] = sliding_window_inference(
                image, roi_size, sw_batch_size, self.forward
            )
            
            batch = self.move_to(batch, 'cpu')

            batch = [self.post_transforms(b) for b in decollate_batch(batch)]
    #        batch = [self.transforms.inverse(b) for b in decollate_batch(batch)]

            assert len(batch) == 1, f"batch size should be 1, but is {len(batch)}."
                    
            # save prediction results
            pred = batch[0]["pred"]
            pred = pred[0].numpy().astype(np.uint8)     # channel 0
            
            if self.cfg.Model.post_process:
                # post processing (cc3d)
                pred = post_process(pred, num_classes=2)
            
            pred = pred.transpose(2, 1, 0)              # conver to (x,y,z) order for sitk
            pred = sitk.GetImageFromArray(pred)
            image_filename = batch[0]['CTres_meta_dict']['filename_or_obj']
            image = sitk.ReadImage(image_filename)
            pred.CopyInformation(image)
            output_filename = f'{self.cfg.Path.output_path}/PRED.nii.gz'
            print('Writing results with SITK format:', output_filename)
            sitk.WriteImage(pred, str(output_filename))
  
        return pred
