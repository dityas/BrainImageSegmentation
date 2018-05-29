import torch
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader
import logging
import numpy
from sklearn.metrics import f1_score


class Trainer:

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 model,
                 batch_size=32):

        self.device = "cuda:0"

        self.batch_size = batch_size
        self.train_dataset = DataLoader(train_dataset,
                                        shuffle=False,
                                        batch_size=self.batch_size,
                                        num_workers=1)

        self.val_dataset = DataLoader(val_dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      num_workers=1)

        self.test_dataset = DataLoader(test_dataset,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       num_workers=1)

        self.model = model
        self.model.to(self.device)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        weight = torch.Tensor([1., 2.]).to(self.device)
        self.loss = N.CrossEntropyLoss(weight=weight)
        self.optimizer = O.Adagrad(params=self.model.parameters())
        self.info_printer = InfoPrinter()

    def dice_coeff(self, prediction, labels):
        """
            Computes dice coefficient.
        """
        # inter = numpy.sum(numpy.dot(labels.ravel(), prediction.ravel()))
        # union = numpy.sum(labels.ravel() + prediction.ravel())
        # dice = ( 2 * inter / (union + 0.00001))
        dice = f1_score(y_true=labels.ravel(), y_pred=prediction.ravel(),
                        average=None,
                        labels=[0, 1])

        return dice

    def run_val_loop(self):
        """
            Runs a loop over the validation dataset and returns the mean
            loss.
        """
        losses = []
        predictions = []
        labels = []
        # Put model in evaluation mode.
        self.model.eval()

        # Run validation loop
        for k, vsample in enumerate(self.val_dataset):
            _in, _out = vsample

            # # Pad inputs and labels to fix convolutions.
            # _in = F.pad(_in, (0, 0, 0, 0, 0, 5), value=0)
            # _out = F.pad(_out, (0, 0, 0, 0, 0, 5), value=0)

            # Move tensors to GPU
            _in = _in.to(self.device).float()
            _out = _out.to(self.device).long()

            # Run prediction loop
            prediction = self.model(_in)
            del _in

            # Report loss and backprop.
            loss = self.loss(prediction,
                             _out)

            prediction = F.log_softmax(prediction,
                                       dim=1).data.cpu().numpy()

            prediction = numpy.argmax(prediction, axis=1)
            label = _out.data.cpu().view(_out.size()[0], -1).numpy()

            losses.append(loss.item())
            predictions.append(prediction)
            labels.append(label)

        # Return to training mode for further training.
        self.model.train()

        # Return mean loss on validation set.
        predictions = numpy.concatenate(predictions, axis=0)
        labels = numpy.concatenate(labels, axis=0)

        dice = self.dice_coeff(prediction=predictions,
                               labels=labels)
        return [numpy.mean(numpy.array(losses)),
                dice]

    def train(self, epochs=10, track_every=20):
        # Put model in training mode.
        self.model.train()
        self.logger.info("Model put in training mode.")

        # Training loop
        for i in range(epochs):

            for j, sample in enumerate(self.train_dataset):
                _in, _out = sample

                # Pad inputs and labels to fix convolutions.
                # _in = F.pad(_in, (0, 0, 0, 0, 0, 5), value=0)
                # _out = F.pad(_out, (0, 0, 0, 0, 0, 5), value=0)

                # Move tensors to GPU
                _in = _in.to(self.device).float()
                _out = _out.to(self.device).long()

                # Run prediction loop
                prediction = self.model(_in)
                del _in

                # Report loss and backprop.
                loss = self.loss(prediction,
                                 _out)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Create metrics report.
                report = {"training_loss": loss.item()}
                          #"dice": dice,
                          #"val_loss": val_loss}

                self.info_printer.print_step_info(report=report,
                                                  epoch=i,
                                                  batch=j,
                                                  print_every=5)

                if j % track_every == 0 and j != 0:

                    print()
                    val_metrics = self.run_val_loop()

                    report = {"dice": val_metrics[1],
                              "val_loss": val_metrics[0]}

                    self.info_printer.line_print(report=report)


class InfoPrinter:
    """
        Prints out training progress and metrics info.
    """

    def __init__(self):
        self.batch = 0
        self.epoch = 0

    def line_print(self, report):
        print()
        metrics = ""

        for key in report.keys():
            metrics += f"{key}: {report[key]} "

        print(f"{metrics}")

        print()

    def print_step_info(self, report, epoch, batch, print_every=50):

        if epoch != self.epoch or ((batch % print_every) == 0):
            print()
            self.epoch = epoch

        metrics = ""

        for key in report.keys():
            metrics += f"{key}: {report[key]} "

        print(f"Epoch: {epoch} Batch: {batch} {metrics}", end='\r')
