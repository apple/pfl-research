# Copyright © 2024 Apple Inc.
import torch
from compare_utils.pytorch import simple_cnn
from core.model import BaseModel
from torch import nn


class SimpleCNN(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = simple_cnn((32, 32, 3), num_outputs=10, transpose=False)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        criterion = nn.CrossEntropyLoss().to(device)
        return criterion(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output,
                                            dim=1) == labels).float()).item()

        return {'output': output, 'acc': accuracy, 'batch_size': n_samples}
