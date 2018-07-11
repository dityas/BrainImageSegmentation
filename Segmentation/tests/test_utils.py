import unittest
import numpy
from pathlib import Path
import sys


path = Path("../code")
sys.path.append(str(path))

from tools.early_stopper import EarlyStopper


class TestScaler(unittest.TestCase):

    def setUp(self):
        self.good_data = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.bad_data = [9, 8, 7, 6, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4, 5, 6]

    def test_early_stopper(self):
        stopper = EarlyStopper()

        for loss in self.good_data:
            print(f"{loss} : {stopper.check_stop_training(loss)}")

        for loss in self.bad_data:
            print(f"{loss} : {stopper.check_stop_training(loss)}")

if __name__ == "__main__":
    unittest.main()
