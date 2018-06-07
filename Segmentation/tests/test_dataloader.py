import unittest
from pathlib import Path
import sys


path = Path("../code")
sys.path.append(str(path))

from dataloader import Dataset2d


class TestDataset(unittest.TestCase):

    def setUp(self):
        data_dir = Path("../data")
        self.data_files = list((data_dir/"HGG").iterdir()) + \
                          list((data_dir/"LGG").iterdir())

        self.dataset = Dataset2d(filenames=self.data_files)

    def test_dataset_creation(self):
        self.assertTrue(len(self.dataset) > 0)

    def test_dataset_iteration(self):
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            print(f"Running sample {i} of {len(self.dataset)}", end='\r')
