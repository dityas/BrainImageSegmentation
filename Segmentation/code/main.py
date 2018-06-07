from pathlib import Path
from dataloader import Dataset2d
from transforms import MinMaxScaler
from trainer import Trainer
from unet2d import UNet2d
import logging


logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

train_files = data_files[:250]
val_files = data_files[251: 253]
test_files = data_files[254:]

train_dataset = Dataset2d(filenames=train_files, name="TrainSet",
                          transform=MinMaxScaler())
val_dataset = Dataset2d(filenames=val_files, name="ValidationSet",
                        transform=MinMaxScaler())
test_dataset = Dataset2d(filenames=test_files, name="TestSet",
                         transform=MinMaxScaler())

trainer = Trainer(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  test_dataset=test_dataset,
                  model=UNet2d(),
                  batch_size=64)

trainer.train(epochs=20, track_every=100)
