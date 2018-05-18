import torch
import torch.nn.functional as F
import torch.optim as O
import torch.autograd as A
import logging


class Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def train(self, epochs=10):
        # Put model in training mode.
        self.model.train()
        self.logger.info("Model put in training mode.")

        # Training loop
        for i in range(epochs):
            for sample in range(len(self.dataset)):
                _sample = self.dataset[sample]

                _in, _out = _sample
                t_in, t_out = torch.from_numpy(_in), torch.from_numpy(_out)

                prediction = self.model(A.Variable(t_in))
                print(prediction)

                break
            break
