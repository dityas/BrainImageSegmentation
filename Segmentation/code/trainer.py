import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader
import logging


class Trainer:

    def __init__(self, dataset, model):
        self.device = "cuda:0"
        self.dataset = DataLoader(dataset,
                                  shuffle=True,
                                  batch_size=1,
                                  num_workers=1)
        self.model = model
        self.model.to(self.device)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.loss = N.BCEWithLogitsLoss()
        self.optimizer = O.Adagrad(params=self.model.parameters())

    def train(self, epochs=10):
        # Put model in training mode.
        self.model.train()
        self.logger.info("Model put in training mode.")

        # Training loop
        for i in range(epochs):
            for j, sample in enumerate(self.dataset):
                _in, _out = sample
                self.optimizer.zero_grad()

                # Pad inputs and labels to fix convolutions.
                _in = F.pad(_in, (0, 0, 0, 0, 0, 5), value=0)
                _out = F.pad(_out, (0, 0, 0, 0, 0, 5), value=0)

                # Move tensors to GPU
                _in = _in.to(self.device).float()
                _out = _out.to(self.device).float()

                # Run prediction loop
                prediction = self.model(_in)
                print(prediction.size())
                #del _in

                # Report loss and backprop.
                #loss = self.loss(prediction.view(-1), _out.view(-1))
                #self.logger.info(f"Epoch: {i} Batch: {j} Loss: {loss.data[0]}")

                #loss.backward()
                #self.optimizer.step()

                break
