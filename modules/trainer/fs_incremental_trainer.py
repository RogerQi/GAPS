import os
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm, trange
from backbone.deeplabv3_renorm import BatchRenorm2d

import utils

from .sequential_GIFS_seg_trainer import sequential_GIFS_seg_trainer

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class kd_criterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_pred, x_ref):
        # x_pred and x_ref are unnormalized
        x_pred = F.softmax(x_pred, dim=1)
        x_ref = F.softmax(x_ref, dim=1)
        element_wise_loss = x_ref * torch.log(x_pred)
        return -torch.mean(element_wise_loss)

def encoder_forward(backbone_net, x):
    x = backbone_net.conv1(x)
    x = backbone_net.bn1(x)
    x = backbone_net.relu(x)
    x = backbone_net.maxpool(x)

    x = backbone_net.layer1(x)
    x = backbone_net.layer2(x)
    x = backbone_net.layer3(x)
    x = backbone_net.layer4(x)

    return F.adaptive_avg_pool2d(x, 1)

memory_bank_size = 500

class fs_incremental_trainer(sequential_GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(fs_incremental_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.partial_data_pool = {}
        self.blank_bank = {}
        self.demo_pool = {}

        self.train_set_vanilla_label = dataset_module.get_train_set_vanilla_label(cfg)

        self.context_aware_prob = self.cfg.TASK_SPECIFIC.GIFS.context_aware_sampling_prob

        assert self.context_aware_prob >= 0
        assert self.context_aware_prob <= 1
    
    def construct_baseset(self):
        baseset_type = self.cfg.TASK_SPECIFIC.GIFS.baseset_type
        baseset_folder = f"save_{self.cfg.name}"
        if baseset_type == 'random':
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            examplar_list = np.arange(0, len(self.train_set))
        elif baseset_type in ['far', 'close', 'far_close', 'uniform_interval', 'class_random']:
            print(f"construct {baseset_type} baseset for {self.cfg.name}")
            base_id_list = self.train_set.dataset.get_label_range()
            if not os.path.exists(f"{baseset_folder}/similarity_dic"):
                self.prv_backbone_net = deepcopy(self.backbone_net)
                self.prv_post_processor = deepcopy(self.post_processor)
                
                self.prv_backbone_net.eval()
                self.prv_post_processor.eval()

                print(f"self.base_id_list contains {base_id_list}")
                base_id_set = set(base_id_list)
                
                mean_weight_dic = {}
                similarity_dic = {}

                # Get the mean weight of each class
                for c in base_id_list:
                    mean_weight_dic[c] = self.prv_post_processor.pixel_classifier.class_mat.weight.data[c]
                    mean_weight_dic[c] = mean_weight_dic[c].reshape((-1))
                    mean_weight_dic[c] = mean_weight_dic[c].cpu().unsqueeze(0)
                    similarity_dic[c] = []
 
                # Maintain a m-size heap to store the top m images of each class
                for i in tqdm(range(len(self.train_set))):
                    img, mask = self.train_set[(i, {'aug': False})]
                    class_list = torch.unique(mask).tolist()
                    img_tensor = torch.stack([img]).to(self.device)
                    mask_tensor = torch.stack([mask]).to(self.device)
                    for c in class_list:
                        if c not in base_id_set:
                            continue
                        with torch.no_grad():
                            img_feature = self.prv_backbone_net(img_tensor)
                            img_weight = utils.masked_average_pooling(mask_tensor == c, img_feature, True)
                        img_weight = img_weight.cpu().unsqueeze(0)
                        similarity = F.cosine_similarity(img_weight, mean_weight_dic[c])
                        ##########################################
                        pixel_count = mask.shape[0] * mask.shape[1]
                        class_pixel_count = torch.sum(mask_tensor == c)
                        ratio = class_pixel_count / pixel_count
                        similarity = similarity.to(self.device)
                        score = similarity / ratio
                        ##########################################
                        similarity_dic[c].append((score, i))
                for c in base_id_list:
                    similarity_dic[c] = sorted(similarity_dic[c], key=lambda x:x[0])
                if not os.path.exists(baseset_folder):   
                    os.makedirs(baseset_folder)  
                torch.save(similarity_dic, f"{baseset_folder}/similarity_dic")
            else:
                print("load similarity_dic")
                similarity_dic = torch.load(f"{baseset_folder}/similarity_dic")
            m = (memory_bank_size // len(base_id_list)) * 2
            if baseset_type == 'far':
                m_far = m
                m_close = 0
            elif baseset_type == 'close':
                m_far = 0
                m_close = m
            elif baseset_type == 'far_close':
                m_close = m // 2
                m_far = m // 2
            elif baseset_type == 'uniform_interval':
                m_interval = m
            elif baseset_type == 'class_random':
                m_interval = m
            else:
                raise NotImplementedError
            # Combine all the top images of each class by set union
            examplar_set = set()
            for c in base_id_list:
                if baseset_type == 'close' or baseset_type == 'far_close':
                    assert m_close <= len(similarity_dic[c])
                    close_list = similarity_dic[c][-m_close:]
                    class_examplar_list = [i for _, i in close_list]
                    examplar_set = examplar_set.union(class_examplar_list)
                if baseset_type == 'far' or baseset_type == 'far_close':
                    assert m_far <= len(similarity_dic[c])
                    far_list = similarity_dic[c][:m_far]
                    class_examplar_list = [i for _, i in far_list]
                    examplar_set = examplar_set.union(class_examplar_list)
                if baseset_type == 'uniform_interval':
                    if len(similarity_dic[c]) < m_interval:
                        print(f"Need {m_interval} from {len(similarity_dic[c])}")
                    local_size = min(m_interval, len(similarity_dic[c]))
                    interval_size = len(similarity_dic[c]) // local_size
                    class_examplar_list = []
                    for i in range(local_size):
                        int_start = interval_size * i
                        assert int_start < len(similarity_dic[c])
                        int_end = min(interval_size * (i + 1), len(similarity_dic[c]))
                        sample_tuple = similarity_dic[c][np.random.randint(low=int_start, high=int_end)]
                        class_examplar_list.append(sample_tuple[1]) # first element is similarity
                    examplar_set = examplar_set.union(class_examplar_list)
                if baseset_type == 'class_random':
                    assert m_interval <= len(similarity_dic[c])
                    selected_id_list = np.random.choice(np.arange(len(similarity_dic[c])), size=(m_interval,), replace=False)
                    class_examplar_list = []
                    for selected_id in selected_id_list:
                        class_examplar_list.append(similarity_dic[c][selected_id][1])
                    examplar_set = examplar_set.union(class_examplar_list)
            examplar_list = sorted(list(examplar_set))
        elif baseset_type == 'RFS':
            base_id_list = self.train_set.dataset.get_label_range()
            t = 0.01
            # count frequency
            total_img_cnt = len(self.train_set)
            category_rf = {}
            for c in base_id_list:
                if c == 0 or c == -1: continue
                sample_cnt = len(self.train_set.dataset.get_class_map(c))
                freq = sample_cnt / total_img_cnt
                category_rf[c] = max(1, math.sqrt(t / freq))

            # Reverse dict to look up class in an image
            sample_factor_arr = np.zeros((len(self.train_set,)))
            for c in base_id_list:
                class_map = self.train_set.dataset.get_class_map(c)
                class_factor = category_rf[c]
                for sample_idx in class_map:
                        sample_factor_arr[sample_idx] = max(sample_factor_arr[sample_idx], class_factor)
            
            # Sample 
            sample_factor_arr = sample_factor_arr / np.sum(sample_factor_arr)
            examplar_list = np.random.choice(np.arange(len(self.train_set)),
                                    replace=False, size=(memory_bank_size,), p=sample_factor_arr)
        elif baseset_type == 'greedy_class_num':
            base_id_list = self.train_set.dataset.get_label_range()
            # Reverse dict to look up class in an image
            sample_cls_arr = np.zeros((len(self.train_set,)))
            for c in base_id_list:
                class_map = self.train_set.dataset.get_class_map(c)
                for sample_idx in class_map:
                    sample_cls_arr[sample_idx] += 1
            
            m = (memory_bank_size // len(base_id_list)) * 3

            examplar_set = set()
            for c in base_id_list:
                cls_sample_list = self.train_set.dataset.get_class_map(c)
                cls_sample_list = np.array(cls_sample_list)
                sample_freq = sample_cls_arr[cls_sample_list]
                sorted_idx = np.argsort(sample_freq)
                selected_idx = sorted_idx[-m:]
                selected_samples = cls_sample_list[selected_idx]
                examplar_set = examplar_set.union(list(selected_samples))
            examplar_list = list(examplar_set)
        else:
            raise AssertionError('invalid baseset_type', baseset_type)
        print(f"total number of examplar_list {len(examplar_list)}")
        return np.random.choice(examplar_list, replace=False, size=(memory_bank_size,))
    
    def test_one(self, device, num_runs=5):
        if self.cfg.TASK_SPECIFIC.GIFS.num_runs != -1:
            num_runs = self.cfg.TASK_SPECIFIC.GIFS.num_runs
        self.base_img_candidates = self.construct_baseset()
        if self.context_aware_prob > 0:
            self.scene_model_setup()
        sequential_GIFS_seg_trainer.test_one(self, device, num_runs)
    
    def synthesizer_sample(self, novel_obj_id):
        # Sample an image from base memory bank
        if torch.rand(1) < self.context_aware_prob:
            # Context-aware sampling from a contextually-similar subset of the memory replay buffer
            memory_buffer_idx = np.random.choice(self.context_similar_map[novel_obj_id])
            base_img_idx = self.base_img_candidates[memory_buffer_idx]
        else:
            # Uniformly sample from the memory buffer
            base_img_idx = np.random.choice(self.base_img_candidates)
        assert base_img_idx in self.base_img_candidates
        assert len(self.base_img_candidates) == memory_bank_size
        syn_img_chw, syn_mask_hw = self.train_set_vanilla_label[base_img_idx]
        # Sample from partial data pool
        # Compute probability for synthesis
        candidate_classes = [c for c in self.partial_data_pool.keys() if c != novel_obj_id]
        # Gather some useful numbers
        num_base_classes = len(self.train_set_vanilla_label.dataset.get_label_range())
        num_novel_classes = len(self.partial_data_pool)
        total_classes = num_base_classes + num_novel_classes
        num_novel_instances = len(self.partial_data_pool[novel_obj_id])
        for k in candidate_classes:
            assert len(self.partial_data_pool[k]) == num_novel_instances, "every class is expected to have $numShot$ samples"
        if self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'vRFS':
            t = 1. / total_classes
            f_n = num_novel_instances / memory_bank_size
            r_e = max(1, np.sqrt(t / f_n))
            r_n = r_e
            other_prob = (r_e + r_n) / (r_e + r_n + 1 + r_n)
            selected_novel_prob = (r_n + r_n) / (r_e + r_n + 1 + r_n)
            num_existing_objects = 2
            num_novel_objects = 2
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'always':
            other_prob = 1
            selected_novel_prob = 1
            num_existing_objects = 2
            num_novel_objects = 2
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'CAS':
            other_prob = 1. / total_classes
            selected_novel_prob = 1. / total_classes
            num_existing_objects = 1
            num_novel_objects = 1
        elif self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat == 'HALF_HALF':
            other_prob = 0.5
            selected_novel_prob = 0.5
            num_existing_objects = 1
            num_novel_objects = 1
        else:
            raise NotImplementedError("Unknown probabilistic synthesis strategy: {}".format(self.cfg.TASK_SPECIFIC.GIFS.probabilistic_synthesis_strat))
        assert other_prob >= 0 and other_prob <= 1
        assert selected_novel_prob >= 0 and selected_novel_prob <= 1

        # Synthesize some other objects other than the selected novel object
        if len(self.partial_data_pool) > 1 and torch.rand(1) < other_prob:
            # select an old class
            for i in range(num_existing_objects):
                selected_class = np.random.choice(candidate_classes)
                selected_sample = random.choice(self.partial_data_pool[selected_class])
                full_img_chw, full_mask_hw, img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = utils.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, selected_class)

        # Synthesize selected novel class
        if torch.rand(1) < selected_novel_prob:
            for i in range(num_novel_objects):
                selected_sample = random.choice(self.partial_data_pool[novel_obj_id])
                full_img_chw, full_mask_hw, img_chw, mask_hw = selected_sample
                syn_img_chw, syn_mask_hw = utils.copy_and_paste(img_chw, mask_hw, syn_img_chw, syn_mask_hw, novel_obj_id)

        return (syn_img_chw, syn_mask_hw)
    
    def get_scene_embedding(self, img):
        '''
        img: normalized image tensor of shape CHW
        '''
        assert len(img.shape) == 3 # CHW
        img = img.view((1,) + img.shape)
        with torch.no_grad():
            scene_embedding = encoder_forward(self.backbone_net, img)
            scene_embedding = scene_embedding.squeeze()
            norm = torch.norm(scene_embedding, p=2)
            scene_embedding = scene_embedding.div(norm+ 1e-5)
            return scene_embedding
    
    def scene_model_setup(self):
        # Compute feature vectors for data in the pool
        self.base_pool_cos_embeddings = []
        for base_data_idx in self.base_img_candidates:
            img_chw, _ = self.train_set[(base_data_idx, {'aug': False})]
            img_chw = img_chw.to(self.device)
            scene_embedding = self.get_scene_embedding(img_chw)
            assert len(scene_embedding.shape) == 1
            self.base_pool_cos_embeddings.append(scene_embedding)
        self.base_pool_cos_embeddings = torch.stack(self.base_pool_cos_embeddings)

    def finetune_backbone(self, base_class_idx, novel_class_idx, support_set):
        assert self.prv_backbone_net is not None
        assert self.prv_post_processor is not None

        scaler = torch.cuda.amp.GradScaler()

        print(f"Adapting to novel id {novel_class_idx}")

        # Learned novel classes
        prv_novel_class_idx = [k for k in support_set.keys() if int(k) < int(min(novel_class_idx))]

        def extract_coco_w_single_instance(ds_reader, sample_idx, class_id, aug_flag):
            # Get raw image and mask
            img_id = ds_reader.dataset.img_ids[sample_idx]
            raw_img = ds_reader.dataset._get_img(img_id)
            raw_mask = ds_reader.dataset._get_mask(img_id)
            assert class_id in raw_mask
            # Get single-instance mask
            img_metadata = ds_reader.dataset.coco.loadImgs(img_id)[0]
            annotations = ds_reader.dataset.coco.imgToAnns[img_id]
            annotations = sorted(annotations, key = lambda x : x['id'])
            seg_mask = None

            for ann in annotations:
                real_class_id = ds_reader.dataset.class_map[ann['category_id']]
                if real_class_id == class_id:
                    ann_mask = torch.from_numpy(ds_reader.dataset.coco.annToMask(ann))
                    assert real_class_id > 0 and real_class_id <= 80
                    seg_mask = torch.zeros((img_metadata['height'], img_metadata['width']), dtype=torch.int64)
                    seg_mask = torch.max(seg_mask, ann_mask * real_class_id)
                    break
            if seg_mask is None:
                raise IndexError(f'Sample idx does not contain {class_id}')

            # Mask out
            seg_mask = seg_mask.int()
            assert seg_mask.shape == raw_mask.shape
            raw_mask[raw_mask == class_id] = 0
            raw_mask[seg_mask == class_id] = class_id

            # Augment if needed
            if aug_flag:
                data = ds_reader.train_data_transforms(raw_img)
                data, label = ds_reader.train_joint_transforms(data, raw_mask)
            else:
                data = ds_reader.test_data_transforms(raw_img)
                data, label = ds_reader.test_joint_transforms(data, raw_mask)
            
            return data, label

        if self.context_aware_prob > 0:
            for novel_obj_id in novel_class_idx:
                assert novel_obj_id in support_set
                for idx in support_set[novel_obj_id]:
                    novel_img_chw, mask_hw = self.continual_train_set[(idx, {'aug': False})]
                    
                    # Compute cosine embedding
                    scene_embedding = self.get_scene_embedding(novel_img_chw.to(self.device))
                    scene_embedding = scene_embedding.view((1,) + scene_embedding.shape)
                    similarity_score = F.cosine_similarity(scene_embedding, self.base_pool_cos_embeddings)
                    base_candidates = torch.argsort(similarity_score)[-int(0.05 * self.base_pool_cos_embeddings.shape[0]):] # Indices array
                    if novel_obj_id not in self.context_similar_map:
                        self.context_similar_map[novel_obj_id] = list(base_candidates.cpu().numpy())
                    else:
                        self.context_similar_map[novel_obj_id] += list(base_candidates.cpu().numpy())
            for c in self.context_similar_map:
                self.context_similar_map[c] = list(set(self.context_similar_map[c]))

        for novel_obj_id in novel_class_idx:
            assert novel_obj_id in support_set
            for idx in support_set[novel_obj_id]:
                novel_img_chw, mask_hw = self.continual_train_set[(idx, {'aug': False})]
                if False:
                    # use single instance
                    novel_img_chw, mask_hw = extract_coco_w_single_instance(self.continual_train_set, idx, novel_obj_id, False)
                img_roi, mask_roi = utils.crop_partial_img(novel_img_chw, mask_hw, cls_id=novel_obj_id)
                assert mask_roi.shape[0] > 0 and mask_roi.shape[1] > 0
                # Minimum bounding rectangle computed; now register it to the data pool
                if novel_obj_id not in self.partial_data_pool:
                    self.partial_data_pool[novel_obj_id] = []
                self.partial_data_pool[novel_obj_id].append((novel_img_chw, mask_hw == novel_obj_id, img_roi, mask_roi))

        self.backbone_net.train()
        self.post_processor.train()

        trainable_params = [
            {"params": self.backbone_net.parameters()},
            {"params": self.post_processor.parameters(), "lr": self.cfg.TASK_SPECIFIC.GIFS.classifier_lr}
        ]

        # Freeze batch norm statistics
        cnt = 0
        for module in self.backbone_net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchRenorm2d):
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
                module.eval()
                cnt += 1
        
        print("Froze {} BN/BRN layers".format(cnt))

        optimizer = optim.SGD(trainable_params, lr = self.cfg.TASK_SPECIFIC.GIFS.backbone_lr, momentum = 0.9)
        
        max_iter = self.cfg.TASK_SPECIFIC.GIFS.max_iter
        def polynomial_schedule(epoch):
            return (1 - epoch / max_iter)**0.9
        batch_size = self.cfg.TASK_SPECIFIC.GIFS.ft_batch_size

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, polynomial_schedule)

        l2_criterion = nn.MSELoss()

        with trange(1, max_iter + 1, dynamic_ncols=True) as t:
            for iter_i in t:
                image_list = []
                mask_list = []
                fully_labeled_flag = []
                partial_positive_idx = []
                for _ in range(batch_size):
                    if True:
                        # synthesis
                        novel_obj_id = random.choice(novel_class_idx)
                        img_chw, mask_hw = self.synthesizer_sample(novel_obj_id)
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                        fully_labeled_flag.append(True)
                    else:
                        if True:
                            # full mask
                            chosen_cls = random.choice(list(novel_class_idx))
                        else:
                            if len(prv_novel_class_idx) > 0:
                                chosen_cls = random.choice(list(prv_novel_class_idx))
                            else:
                                chosen_cls = random.choice(list(novel_class_idx))
                        idx = random.choice(support_set[chosen_cls])
                        img_chw, mask_hw = self.continual_train_set[idx]
                        if False:
                            # Mask non-novel portion using pseudo labels
                            with torch.no_grad():
                                data_bchw = img_chw.to(self.device)
                                data_bchw = data_bchw.view((1,) + data_bchw.shape)
                                feature = self.prv_backbone_net(data_bchw)
                                ori_spatial_res = data_bchw.shape[-2:]
                                output = self.prv_post_processor(feature, ori_spatial_res, scale_factor=10)
                            tmp_target_hw = output.max(dim = 1)[1].cpu()[0]
                            novel_mask = torch.zeros_like(mask_hw)
                            for novel_obj_id in support_set.keys():
                                novel_mask = torch.logical_or(novel_mask, mask_hw == novel_obj_id)
                            tmp_target_hw[novel_mask] = mask_hw[novel_mask]
                            mask_hw = tmp_target_hw
                        if False:
                            # use single instance
                            img_chw, mask_hw = extract_coco_w_single_instance(self.continual_train_set, idx, chosen_cls, True)
                        # mask_hw[mask_hw != chosen_cls] = 0
                        # mask_hw[mask_hw == chosen_cls] = 1
                        image_list.append(img_chw)
                        mask_list.append(mask_hw)
                        fully_labeled_flag.append(True)
                        # partial_positive_idx.append(chosen_cls)
                partial_positive_idx = torch.tensor(partial_positive_idx)
                fully_labeled_flag = torch.tensor(fully_labeled_flag)
                data_bchw = torch.stack(image_list).to(self.device).detach()
                target_bhw = torch.stack(mask_list).to(self.device).detach().long()

                with torch.cuda.amp.autocast(enabled=True):
                    feature = self.backbone_net(data_bchw)
                    ori_spatial_res = data_bchw.shape[-2:]
                    output = self.post_processor(feature, ori_spatial_res, scale_factor=10)

                    if True:
                        # CE loss on all fully-labeled images
                        output_logit_bchw = F.softmax(output, dim=1)
                        output_logit_bchw = torch.log(output_logit_bchw)
                        loss = F.nll_loss(output_logit_bchw[fully_labeled_flag], target_bhw[fully_labeled_flag], ignore_index=-1)

                        # MiB loss (modified CE) on partially-labeled images
                        if len(partial_positive_idx) > 0:
                            partial_labeled_flag = torch.logical_not(fully_labeled_flag)
                            output_logit_bchw = F.softmax(output[partial_labeled_flag], dim=1)
                            # Reduce to 0/1 using MiB for NLL loss
                            assert output_logit_bchw.shape[0] == len(partial_positive_idx)
                            fg_prob_list = []
                            bg_prob_list = []
                            for b in range(output_logit_bchw.shape[0]):
                                fg_prob_hw = output_logit_bchw[b,partial_positive_idx[b]]
                                bg_prob_chw = output_logit_bchw[b,torch.arange(output_logit_bchw.shape[1]) != partial_positive_idx[b]]
                                bg_prob_hw = torch.sum(bg_prob_chw, dim=0)
                                fg_prob_list.append(fg_prob_hw)
                                bg_prob_list.append(bg_prob_hw)
                            fg_prob_map = torch.stack(fg_prob_list)
                            bg_prob_map = torch.stack(bg_prob_list)
                            reduced_output_bchw = torch.stack([bg_prob_map, fg_prob_map], dim=1)
                            reduced_logit_bchw = torch.log(reduced_output_bchw)
                            loss = loss + F.nll_loss(reduced_logit_bchw, target_bhw[partial_labeled_flag], ignore_index=-1)
                    else:
                        loss = self.criterion(output, target_bhw)

                    # L2 regularization on feature extractor
                    with torch.no_grad():
                        # self.vanilla_backbone_net for the base version
                        ori_feature = self.prv_backbone_net(data_bchw)
                        # ori_logit = self.prv_post_processor(ori_feature, ori_spatial_res, scale_factor=10)

                    # Feature extractor regularization + classifier regularization
                    regularization_loss = l2_criterion(feature, ori_feature)
                    regularization_loss = regularization_loss * self.cfg.TASK_SPECIFIC.GIFS.feature_reg_lambda # hyperparameter lambda
                    loss = loss + regularization_loss

                optimizer.zero_grad() # reset gradient
                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update()
                scheduler.step()
                t.set_description_str("Loss: {:.4f}".format(loss.item()))
