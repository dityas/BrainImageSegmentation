from pathlib import Path
from dataloader import fMRIDataset
from trainer import Trainer
from model import UNet, LameCNN
import logging


logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

train_files = data_files[:200]
val_files = data_files[200:215]
test_files = data_files[215:]

train_dataset = fMRIDataset(filenames=train_files, name="TrainSet")
val_dataset = fMRIDataset(filenames=val_files, name="ValidationSet")
test_dataset = fMRIDataset(filenames=test_files, name="TestSet")

trainer = Trainer(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  test_dataset=test_dataset,
                  model=UNet(),
                  batch_size=2)

trainer.train(epochs=20)
