from pathlib import Path
from dataloader import fMRIDataset
import logging


logging.basicConfig(level=logging.DEBUG)
data_dir = Path("../data")
data_files = list((data_dir/"HGG").iterdir()) + \
             list((data_dir/"LGG").iterdir())

dataset = fMRIDataset(filenames=data_files, name="Dataset1")

for i in range(len(dataset)):
    sample = dataset[i]
    _in, _out = sample
    print(f"Sample {i} has input {_in.shape} and output {_out.shape}")
