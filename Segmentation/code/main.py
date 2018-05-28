from pathlib import Path
from dataloader import T1Dataset2d
from trainer import Trainer
from unet2d import UNet2d
import logging


logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

train_files = data_files[:210]
val_files = data_files[210:215]
test_files = data_files[215:]

train_dataset = T1Dataset2d(filenames=train_files, name="TrainSet")
val_dataset = T1Dataset2d(filenames=val_files, name="ValidationSet")
test_dataset = T1Dataset2d(filenames=test_files, name="TestSet")

trainer = Trainer(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  test_dataset=test_dataset,
                  model=UNet2d(),
                  batch_size=16)

trainer.train(epochs=10)
