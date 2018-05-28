from pathlib import Path
from dataloader import T1Dataset2d
from trainer import Trainer
from model import LameCNN
import logging


logging.basicConfig(level=logging.DEBUG)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

train_files = data_files[:200]
val_files = data_files[200:215]
test_files = data_files[215:]

train_dataset = T1Dataset2d(filenames=train_files, name="TrainSet")
val_dataset = T1Dataset2d(filenames=val_files, name="ValidationSet")
test_dataset = T1Dataset2d(filenames=test_files, name="TestSet")

trainer = Trainer(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  test_dataset=test_dataset,
                  model=LameCNN(),
                  batch_size=1)

trainer.train(epochs=1)
