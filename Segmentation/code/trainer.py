import torch
import torch.nn.functional as F
import torch.optim as O
import torch.autograd as A
from torch.utils.data import DataLoader
import logging


class Trainer:

    def __init__(self, dataset, model):
        self.dataset = DataLoader(dataset,
                                  shuffle=True,
                                  batch_size=1,
                                  num_workers=2)
        self.model = model
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.device = "cuda:0"

    def train(self, epochs=10):
        # Put model in training mode.
        self.model.to(self.device)
        self.model.train()
        self.logger.info("Model put in training mode.")

        # Training loop
        for i in range(epochs):
            for i, sample in enumerate(self.dataset):
                _sample = sample

                _in, _out = _sample
                print(_in.shape)
                #t_in, t_out = torch.from_numpy(_in), torch.from_numpy(_out)

                #prediction = self.model(A.Variable(t_in))
                #print(prediction.size())

                break
