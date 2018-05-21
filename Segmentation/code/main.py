from pathlib import Path
from dataloader import fMRIDataset
from trainer import Trainer
from model import UNet, LameCNN
import logging


logging.basicConfig(level=logging.INFO)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

dataset = fMRIDataset(filenames=data_files, name="Dataset1")
trainer = Trainer(dataset=dataset, model=LameCNN())
trainer.train(epochs=10)
