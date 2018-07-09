from pathlib import Path
from dataloader import Dataset2d
from tools.pipeline import SegmentationPipeline
from unet2d import UNet2d
import torch
from torch.utils.data import DataLoader
import logging
import torch.nn as N
import torch.optim as O

logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

train_files = data_files[:250]
val_files = data_files[251: 253]
test_files = data_files[254:]

train_dataset = Dataset2d(filenames=train_files, name="TrainSet")
val_dataset = Dataset2d(filenames=val_files, name="ValidationSet")
test_dataset = Dataset2d(filenames=test_files, name="TestSet")
train_dataset = DataLoader(train_dataset,
                           shuffle=False,
                           batch_size=64,
                           num_workers=1)

val_dataset = DataLoader(val_dataset,
                         shuffle=False,
                         batch_size=64,
                         num_workers=1)

test_dataset = DataLoader(test_dataset,
                          shuffle=True,
                          batch_size=32,
                          num_workers=1)

weight = torch.Tensor([1., 2.])
loss_fn = N.CrossEntropyLoss(weight=weight)

pipeline = SegmentationPipeline(training_set=train_dataset,
                                validation_set=val_dataset,
                                testing_set=test_dataset,
                                loss=loss_fn,
                                model=UNet2d(),
                                optimizer=O.Adagrad,
                                device="cuda:0")

pipeline.train(epochs=10, track_every=5)
