import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
from tqdm import tqdm

import utils

from .trainer_base import trainer_base

class seg_trainer(trainer_base):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)

    def train_one(self, device, optimizer, epoch):
        assert self.device == torch.device('cuda'), "Training support only GPU now"
        scaler = torch.cuda.amp.GradScaler()
        self.backbone_net.train()
        self.post_processor.train()
        start_cp = time.time()
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad() # reset gradient
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                loss = self.criterion(output, target)
            if True:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_total_loss += loss.item()
            if True:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if batch_idx % self.cfg.TRAIN.log_interval == 0:
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only)
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f} Epoch Elapsed Time: {6:.1f}'.format(
                    epoch, batch_idx * len(data), len(self.train_set),
                    100. * batch_idx / len(self.train_loader), loss.item(), batch_acc, time.time() - start_cp))
        
        return train_total_loss / len(self.train_loader)

    def val_one(self, device):
        if self.cfg.meta_training_num_classes != -1:
            class_iou, pixel_acc = self.eval_on_loader(self.val_loader, self.cfg.meta_training_num_classes)
        else:
            class_iou, pixel_acc = self.eval_on_loader(self.val_loader, self.cfg.num_classes)
        print('Test set: Mean IoU {:.4f}'.format(np.mean(class_iou)))
        print('Test set: Mean Acc {:.4f}'.format(pixel_acc))
        print("Class-wise IoU:")
        print(class_iou)
        return np.mean(class_iou)

    def test_one(self, device):
        return self.val_one(device)
    
    def trace_model(self, target_path, normalization=True):
        """Trace a trained model and serialize to a path
        See https://pytorch.org/docs/stable/generated/torch.jit.trace.html

        Args:
            target_path (str): Target path to store serialized model
            normalization (bool, optional): Whether normalization layer is used. When set to
            True, UINT8 images ranging from 0-255 are expected. Defaults to True.
        """
        self.backbone_net.eval()
        self.post_processor.eval()
        dummy_tensor = torch.rand((1,) + self.cfg.input_dim).cuda() # BCHW
        # Function factory
        norm_mean = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
        norm_std = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
        class dummy_predictor(nn.Module):
            def __init__(self, backbone_net, post_processor):
                super(dummy_predictor, self).__init__()
                self.backbone_net_ = backbone_net
                self.post_processor_ = post_processor
            
            def forward(self, input_tensor):
                if normalization:
                    with torch.no_grad():
                        input_tensor = input_tensor / 255.0
                        input_tensor = tv.transforms.functional.normalize(input_tensor, norm_mean, norm_std)
                with torch.no_grad():
                    feature = self.backbone_net_(input_tensor)
                    ori_spatial_res = input_tensor.shape[-2:]
                    output = self.post_processor_(feature, ori_spatial_res)
                    output = torch.softmax(output, dim=1) # BCHW
                    return output
        my_dummy_predictor = dummy_predictor(self.backbone_net, self.post_processor)
        traced_script_module = torch.jit.trace(my_dummy_predictor, dummy_tensor)
        traced_script_module.save(target_path)

    
    def eval_on_loader(self, test_loader, num_classes, visfreq=9999999999, masked_class=None):
        self.backbone_net.eval()
        self.post_processor.eval()
        test_loss = 0
        pixel_acc_list = []
        class_intersection, class_union = (None, None)
        class_names_list = test_loader.dataset.dataset.CLASS_NAMES_LIST
        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(test_loader)):
                data, target = data.to(self.device), target.to(self.device)
                feature = self.backbone_net(data)
                ori_spatial_res = data.shape[-2:]
                output = self.post_processor(feature, ori_spatial_res)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=self.cfg.METRIC.SEGMENTATION.fg_only, masked_class=masked_class)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    pred_np = np.array(pred_map[i].cpu())
                    target_np = np.array(target[i].cpu(), dtype=np.int64)
                    intersection, union = utils.compute_iu(pred_np, target_np, num_classes, masked_class=masked_class)
                    if class_intersection is None:
                        class_intersection = intersection
                        class_union = union
                    else:
                        class_intersection += intersection
                        class_union += union
                    if (idx + 1) % visfreq == 0:
                        gt_label = utils.visualize_segmentation(self.cfg, data[i], target_np, None)
                        predicted_label = utils.visualize_segmentation(self.cfg, data[i], pred_np, None)
                        np.save('/tmp/{}_{}_pred.npy'.format(idx, i), pred_np)
                        # np.save('/tmp/{}_{}_gt.npy'.format(idx, i), target_np)
                        # cv2.imwrite("{}_{}_pred.png".format(idx, i), predicted_label)
                        # cv2.imwrite("{}_{}_label.png".format(idx, i), gt_label)
                        # Visualize RGB image as well
                        ori_rgb_np = np.array(data[i].permute((1, 2, 0)).cpu())
                        if 'normalize' in self.cfg.DATASET.TRANSFORM.TEST.transforms:
                            rgb_mean = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
                            rgb_sd = self.cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
                            ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
                        assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
                        ori_rgb_np[ori_rgb_np >= 1] = 1
                        ori_rgb_np = (ori_rgb_np * 255).astype(np.uint8)
                        # Convert to OpenCV BGR
                        ori_rgb_np = cv2.cvtColor(ori_rgb_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("/tmp/{}_{}_ori.jpg".format(idx, i), ori_rgb_np)

        test_loss /= len(test_loader.dataset)

        class_iou = class_intersection / (class_union + 1e-10)
        mean_acc = np.mean(pixel_acc_list)
        return class_iou, mean_acc
