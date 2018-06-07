import unittest
import numpy
from pathlib import Path
import sys


path = Path("../code")
sys.path.append(str(path))

from dataloader import Dataset2d
from transforms import MinMaxScaler


class TestDataset(unittest.TestCase):

    def setUp(self):
        data_dir = Path("../data")
        self.data_files = list((data_dir/"HGG").iterdir()) + \
                          list((data_dir/"LGG").iterdir())

        self.dataset = Dataset2d(filenames=self.data_files)
        self.scaler = MinMaxScaler()
        self.tdataset = Dataset2d(filenames=self.data_files,
                                  transform=[MinMaxScaler()])

    def test_dataset_creation(self):
        self.assertTrue(len(self.dataset) > 0)

    # def test_dataset_iteration(self):
    #     print()
    #     for i in range(len(self.dataset)):
    #         _ = self.dataset[i]
    #         print(f"Running sample {i} of {len(self.dataset)}", end='\r')
    #     print()
    #
    # def test_dataset_scaling(self):
    #     print()
    #     for i in range(len(self.dataset)):
    #         _in, out = self.dataset[i]
    #         print(f"Checking sample {i} of {len(self.dataset)}", end="\r")
    #         self.scaler.partial_fit(_in)
    #         _in = self.scaler.transform(_in)
    #         self.assertTrue(numpy.max(_in) <= 1.0 and numpy.min(_in) >= 0.0)
    #     print()

    def test_internal_transforms(self):
        print()
        for i in range(len(self.tdataset)):
            _in, out = self.tdataset[i]
            print(f"Checking sample {i} of {len(self.tdataset)}", end="\r")
            self.assertTrue(numpy.max(_in) <= 1.0 and numpy.min(_in) >= 0.0)
        print()


if __name__ == "__main__":
    unittest.main()
