import math
import torch

from abc import abstractmethod

class trainer_base(object):
    def __init__(self, cfg, backbone_net, post_processor, criterion, dataset_module, device):
        self.cfg = cfg
        self.backbone_net = backbone_net
        self.post_processor = post_processor
        self.criterion = criterion
        self.device = device
        self.feature_shape = self.backbone_net.get_feature_tensor_shape(self.device)

        # Obtain dataset
        self.train_set = dataset_module.get_train_set(cfg)
        self.val_set = dataset_module.get_val_set(cfg)

        print("Training set contains {} data points. Val set contains {} data points.".format(len(self.train_set), len(self.val_set)))

        # Prepare loaders
        # TODO: change to iteration-based loader rather than epoch-based.
        self.loader_kwargs = {
            'num_workers': cfg.SYSTEM.num_workers,
            'pin_memory': cfg.SYSTEM.pin_memory,
            'drop_last': True}

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=cfg.TRAIN.batch_size, shuffle=True, **self.loader_kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=cfg.TEST.batch_size, shuffle=False, **self.loader_kwargs)

    @abstractmethod
    def train_one(self, device, optimizer, epoch):
        raise NotImplementedError

    @abstractmethod
    def val_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def test_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def live_run(self, device):
        raise NotImplementedError
    
    def adapt_scheduler(self, start_epoch, scheduler):
        assert isinstance(start_epoch, int) and start_epoch >= 1
        if start_epoch > 1:
            if self.cfg.TRAIN.step_per_iter:
                elapsed_iter = math.ceil(len(self.train_set)  / self.cfg.TRAIN.batch_size) * (start_epoch - 1)
                for _ in range(elapsed_iter):
                    scheduler.step()
            else:
                # Tune LR scheduler
                for _ in range(1, start_epoch):
                    scheduler.step()

    def save_model(self, file_path):
        """Save default model (backbone_net, post_processor to a specified file path)

        Args:
            file_path (str): path to save the model
        """
        torch.save({
            "backbone": self.backbone_net.state_dict(),
            "head": self.post_processor.state_dict()
        }, file_path)
    
    def load_model(self, file_path):
        """Load weights for default model components (backbone_net, post_process) from a given file path

        Args:
            file_path (str): path to trained weights
        """
        trained_weight_dict = torch.load(file_path)
        self.backbone_net.load_state_dict(trained_weight_dict['backbone'], strict=True)
        self.post_processor.load_state_dict(trained_weight_dict['head'], strict=True)
