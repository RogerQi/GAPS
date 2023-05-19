import numpy as np
from copy import deepcopy
from .GIFS_seg_trainer import GIFS_seg_trainer

def harmonic_mean(base_iou, novel_iou):
    return 2 / (1. / base_iou + 1. / novel_iou)

class sequential_GIFS_seg_trainer(GIFS_seg_trainer):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        super(sequential_GIFS_seg_trainer, self).__init__(cfg, backbone_net, post_processor, criterion, dataset_module, device)
        
        self.partial_data_pool = {}
    
    def continual_test_single_pass(self, support_set):
        self.partial_data_pool = {}
        self.context_similar_map = {}
        self.vanilla_backbone_net = deepcopy(self.backbone_net)
        self.vanilla_post_processor = deepcopy(self.post_processor)

        all_novel_class_idx = sorted(list(support_set.keys()))
        base_class_idx = self.train_set.dataset.visible_labels
        if 0 not in base_class_idx:
            base_class_idx.append(0)
        base_class_idx = sorted(base_class_idx)

        self.vanilla_base_class_idx = deepcopy(base_class_idx)
        learned_novel_class_idx = []

        total_num_classes = len(all_novel_class_idx) + len(base_class_idx)

        # Construct task batches
        assert len(all_novel_class_idx) % self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes == 0
        num_tasks = len(all_novel_class_idx) // self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes
        ptr = 0
        task_stream = []
        for i in range(num_tasks):
            current_task = []
            for j in range(self.cfg.TASK_SPECIFIC.GIFS.sequential_dataset_num_classes):
                current_task.append(all_novel_class_idx[ptr])
                ptr += 1
            task_stream.append(current_task)
        assert ptr == len(all_novel_class_idx)

        for i, task in enumerate(task_stream):
            self.prv_backbone_net = deepcopy(self.backbone_net)
            self.prv_post_processor = deepcopy(self.post_processor)
            
            self.prv_backbone_net.eval()
            self.prv_post_processor.eval()
            self.backbone_net.eval()
            self.post_processor.eval()

            if len(base_class_idx) != self.prv_post_processor.pixel_classifier.class_mat.weight.data.shape[0]:
                # squeeze the classifier weights
                self.prv_post_processor.pixel_classifier.class_mat.weight.data = self.prv_post_processor.pixel_classifier.class_mat.weight.data[base_class_idx]

            self.novel_adapt(base_class_idx, task, support_set)
            learned_novel_class_idx += task

            # Evaluation
            # Following PIFS, completely unseen classes are excluded from evaluation
            # https://github.com/fcdl94/FSS/blob/master/metrics/stream_metrics.py#L92
            unseen_classes = [i for i in support_set.keys() if i not in learned_novel_class_idx]
            print(f"Classes {unseen_classes} are masked from current evaluation.")
            classwise_iou, mean_pixel_acc = self.eval_on_loader(self.continual_test_loader, total_num_classes, masked_class=unseen_classes)

            classwise_iou = np.array(classwise_iou)

            # to handle background and 0-indexing
            novel_iou_list = []
            base_iou_list = []
            for i in range(len(classwise_iou)):
                label = i
                if label in learned_novel_class_idx:
                    novel_iou_list.append(classwise_iou[i])
                elif label in self.vanilla_base_class_idx:
                    base_iou_list.append(classwise_iou[i])
                else:
                    continue
            base_iou = np.mean(base_iou_list)
            novel_iou = np.mean(novel_iou_list)

            print("[PIFS-MASKING] Base IoU: {:.4f} Novel IoU: {:.4f}".format(base_iou, novel_iou))
            print("[PIFS-MASKING] Novel class wise IoU: {}".format(novel_iou_list))

            base_class_idx += task
            base_class_idx = sorted(base_class_idx)

        # Restore weights
        self.backbone_net = self.vanilla_backbone_net
        self.post_processor = self.vanilla_post_processor

        return classwise_iou