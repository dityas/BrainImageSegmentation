from pathlib import Path
from dataloader import Dataset2d
from tools.pipeline import SegmentationPipeline
from unet2d import UNet2d
import torch
from torch.utils.data import DataLoader
import logging
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
from random import shuffle


# Specify device
device = "cuda:1"

# Set up datasets
logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())
shuffle(data_files)

train_files = data_files[:250]
val_files = data_files[251: 260]
test_files = data_files[260:]

train_dataset = Dataset2d(filenames=train_files, name="TrainSet")
val_dataset = Dataset2d(filenames=val_files, name="ValidationSet")
test_dataset = Dataset2d(filenames=test_files, name="TestSet")
train_dataset = DataLoader(train_dataset,
                           shuffle=False,
                           batch_size=16,
                           num_workers=1)

val_dataset = DataLoader(val_dataset,
                         shuffle=True,
                         batch_size=64,
                         num_workers=1,
                         drop_last=True)

test_dataset = DataLoader(test_dataset,
                          shuffle=True,
                          batch_size=2,
                          num_workers=1)

# Initialize loss functions.
# weight = torch.Tensor([1., 2.]).to(device)
loss_fn = N.CrossEntropyLoss()


def dice_loss(predictions, targets):
    """
        Computes loss based on combination of BCE and Dice.
    """
    bce_loss = loss_fn(predictions, targets)
    predictions = F.log_softmax(predictions, dim=1).select(1, 1)
    predictions = predictions.contiguous().view(-1).float()
    targets = targets.view(-1).float()

    intersection = torch.dot(predictions, targets).sum()
    union = predictions.sum() + targets.sum()

    dice_score = (2.0 * (intersection + 1) / (union + 1))

    return (0.1 * bce_loss) + (1 - dice_score)


# Define metric.
def dice_score(predictions, targets):
    predictions = predictions.select(1, 1).contiguous().view(-1).float()
    targets = targets.view(-1).float()

    intersection = torch.dot(predictions, targets).sum()
    union = predictions.sum() + targets.sum()

    return (2.0 * (intersection + 1) / (union + 1))


# Run pipeline.
pipeline = SegmentationPipeline(training_set=train_dataset,
                                validation_set=val_dataset,
                                testing_set=test_dataset,
                                loss=dice_loss,
                                model=UNet2d(),
                                optimizer=O.Adagrad,
                                device=device,
                                metric=dice_score,
                                early_stopping_patience=20)

pipeline.train(epochs=1000, track_every=500)
