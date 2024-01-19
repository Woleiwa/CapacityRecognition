import torch
import torch.nn as nn
import torch.nn.utils as utils


class Recorder:
    def __init__(self, model: nn.Module, epoch, optimizer, criterion):
        self.model = model
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion

    def write_to_file(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, path)

    def read_from_file(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.criterion = checkpoint['loss']