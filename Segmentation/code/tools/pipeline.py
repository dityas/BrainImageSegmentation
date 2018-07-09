import logging
from torchvision import transforms
import torch


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
                 metric=None,
                 device="cpu:0"):

        self.training_set = training_set
        self.testing_set = testing_set
        self.validation_set = validation_set
        self.loss = loss
        self.model = model
        self.optimizer = optimizer(params=self.model.parameters())
        self.metric = metric

        if transform_list is not None:
            self.transforms = transforms.Compose(transform_list)
        else:
            self.transforms = None

        self.device = device

        self.model.to(self.device)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def __move_sample_to_device(self, sample):
        """
            Move the tensors to specified device.
        """
        _in, _target = sample
        _in.to(self.device)
        _target.to(self.device)

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

        for j, sample in enumerate(self.validation_set):
            _in, _target = self.__move_sample_to_device(sample)
            losses.append(self.loss(self.model(_in), _target))

        loss = torch.mean(torch.tensor(losses))

        if self.metric is not None:
            metric = self.metric()
        else:
            metric = float('nan')

        return loss, metric

    def print_progress(self,
                       epoch,
                       batch,
                       loss=float('nan'),
                       val_loss=float('nan'),
                       metric=float('nan')):

        banner = f"""
                  Epoch: {epoch} | Batch: {batch}
                  loss: {loss:.5f}
                  val_loss: {val_loss:.5f}
                  metric: {metric:.5f}
                  """

        print(banner, end="\r")

    def train(self,
              epochs=10,
              track_every=20):
        """
            Defines the main training loop of the pipeline.
        """
        self.model.train()
        print("Model put in training mode.")

        for i in range(epochs):
            for j, sample in enumerate(self.training_set):

                # Run single loop.
                loss = self.partial_fit(sample)
                self.print_progress(epoch=i,
                                    batch=j,
                                    loss=loss)

            val_loss, metric = self.run_validation()
            self.print_progress(epoch=i,
                                batch=j,
                                loss=loss,
                                val_loss=val_loss,
                                metric=metric)
