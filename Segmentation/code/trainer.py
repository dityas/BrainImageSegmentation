import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader
import logging
import numpy


class Trainer:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 model):

        self.device = "cpu"

        self.train_dataset = DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=1,
                                        num_workers=1)

        self.val_dataset = DataLoader(val_dataset,
                                      shuffle=True,
                                      batch_size=1,
                                      num_workers=1)

        self.test_dataset = DataLoader(test_dataset,
                                       shuffle=True,
                                       batch_size=1,
                                       num_workers=1)

        self.model = model
        self.model.to(self.device)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        self.loss = N.BCEWithLogitsLoss()
        self.optimizer = O.Adagrad(params=self.model.parameters())
        self.info_printer = InfoPrinter()

    def run_val_loop(self):

        losses = []

        self.model.eval()
        for k, vsample in enumerate(self.val_dataset):
            _in, _out = vsample

            # Pad inputs and labels to fix convolutions.
            _in = F.pad(_in, (0, 0, 0, 0, 0, 5), value=0)
            _out = F.pad(_out, (0, 0, 0, 0, 0, 5), value=0)

            # Move tensors to GPU
            _in = _in.to(self.device).float()
            _out = _out.to(self.device).float()

            # Run prediction loop
            prediction = self.model(_in)
            del _in

            # Report loss and backprop.
            loss = self.loss(prediction.view(-1), _out.view(-1))

            losses.append(loss.item())

        self.model.train()
        return numpy.mean(numpy.array(losses))

    def train(self, epochs=10):
        # Put model in training mode.
        self.model.train()
        self.logger.info("Model put in training mode.")

        # Training loop
        for i in range(epochs):
            for j, sample in enumerate(self.train_dataset):
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
                del _in

                # Report loss and backprop.
                loss = self.loss(prediction.view(-1), _out.view(-1))
                val_loss = self.run_val_loop()

                # Create metrics report.
                report = {"training_loss": loss.item(),
                          "validation_loss": val_loss}

                self.info_printer.print_step_info(report=report,
                                                  epoch=i,
                                                  batch=j)

                loss.backward()
                self.optimizer.step()

                # break


class InfoPrinter:
    """
        Prints out training progress and metrics info.
    """

    def __init__(self):
        self.batch = 0
        self.epoch = 0

    def print_step_info(self, report, epoch, batch):

        if epoch != self.epoch:
            print()
            self.epoch = epoch

        metrics = ""

        for key in report.keys():
            metrics += f"{key}: {report[key]} "

        print(f"Epoch: {epoch} Batch: {batch} {metrics}", end='\r')
