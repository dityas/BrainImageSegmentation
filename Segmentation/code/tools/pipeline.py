import logging
from torchvision import transforms


class SegmentationPipeline:

    """
        Defines a pipeline.
    """
    def __init__(self,
                 training_set,
                 testing_set,
                 validation_set,
                 transform_list=None,
                 model=None,
                 device="cpu:0"):

        self.training_set = training_set
        self.testing_set = testing_set
        self.validation_set = validation_set

        if transform_list is not None:
            self.transforms = transforms.Compose(transform_list)
        else:
            self.transforms = None

        self.model = model
        self.device = device

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def __move_sample_to_device(self, sample):
        """
            Move the tensors to specified device.
        """
        _in, _target = sample
        _in.to(self.device)
        _target.to(self.device)

        return _in, _target

    def train(self,
              epochs=10,
              track_every=20):
        """
            Defines the main training loop of the pipeline.
        """
        for i in range(epochs):
            for j, sample in enumerate(self.training_set):

                # Get batch and move data to devic
                _in, _target = self.__move_sample_to_device(sample)

                # Apply transforms
                if self.transforms is not None:
                    _in, _target = self.transforms(_in, _target)

                
