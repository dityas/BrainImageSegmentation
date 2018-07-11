import logging
from torchvision import transforms
import torch
from tools.early_stopper import EarlyStopper


class SegmentationPipeline:

    """
        Defines a pipeline.
    """
    def __init__(self,
                 training_set,
                 testing_set,
                 validation_set,
                 model,
                 loss,
                 optimizer,
                 transform_list=None,
                 early_stopping_patience=10,
                 metric=None,
                 device="cpu:0"):

        self.training_set = training_set
        self.testing_set = testing_set
        self.validation_set = validation_set
        self.estopper = EarlyStopper(patience=early_stopping_patience)
        self.loss = loss
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer(params=self.model.parameters())
        self.metric = metric

        if transform_list is not None:
            self.transforms = transforms.Compose(transform_list)
        else:
            self.transforms = None

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def __move_sample_to_device(self, sample):
        """
            Move the tensors to specified device.
        """
        _in, _target = sample
        _in = _in.to(self.device).float()
        _target = _target.to(self.device).long()

        return _in, _target

    def partial_fit(self, sample):
        """
            Runs one update of training.
        """
        # Get batch and move data to devic
        _in, _target = self.__move_sample_to_device(sample)

        # Apply transforms
        if self.transforms is not None:
            _in, _target = self.transforms(_in, _target)

        # Run forward loop
        prediction = self.model(_in)

        # Calculate loss
        loss = self.loss(prediction, _target)

        # Update gradients.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_validation(self):

        self.model.eval()
        losses = []
        targets = []
        predictions = []

        for j, sample in enumerate(self.validation_set):
            _in, _target = self.__move_sample_to_device(sample)
            losses.append(self.loss(self.model(_in), _target).item())
            predictions.append(self.model.predict(_in).view(-1))
            targets.append(_target.view(-1))

        loss = torch.mean(torch.tensor(losses))

        if self.metric is not None:
            metric = self.metric(torch.stack(predictions),
                                 torch.stack(targets))
        else:
            metric = float('nan')

        self.model.train()
        return loss, metric

    def print_progress(self,
                       epoch,
                       batch,
                       loss=float('nan'),
                       val_loss=float('nan'),
                       metric=float('nan')):

        banner = f"Epoch: {epoch:3d} | Batch: {batch:3d} "
        banner += f"loss: {loss:2.5f} "
        banner += f"val_loss: {val_loss:2.5f} "
        banner += f"metric: {metric:2.5f} "
        print(banner, end="\r")

    def update_validation_result(self, epoch, batch, loss):
        val_loss, metric = self.run_validation()
        self.print_progress(epoch=epoch,
                            batch=batch,
                            loss=loss,
                            val_loss=val_loss,
                            metric=metric)
        print()

        return val_loss, metric

    def train(self,
              epochs=10,
              track_every=20):
        """
            Defines the main training loop of the pipeline.
        """
        self.model.train()
        print("Model put in training mode.")

        for i in range(epochs):
            stop_training = False
            batch_losses = []
            for j, sample in enumerate(self.training_set):

                # Run single loop.
                loss = self.partial_fit(sample)
                batch_losses.append(loss)
                self.print_progress(epoch=i,
                                    batch=j,
                                    loss=loss)

                if j % track_every == 0 and j != 0:
                    batch_loss = torch.mean(torch.tensor(batch_losses))
                    val_loss, metric = self.update_validation_result(epoch=i,
                                                                     batch=j,
                                                                     loss=batch_loss)

                    stop_training = self.estopper.check_stop_training(val_loss)

                    if stop_training:
                        break

            # End batch iteration.

            val_loss, metric = self.update_validation_result(epoch=i,
                                                             batch=j,
                                                             loss=batch_loss)

            if stop_training:
                print("Early stopping.")
                break

        # End training loop.
