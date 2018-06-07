import unittest
import numpy
from pathlib import Path
import sys


path = Path("../code")
sys.path.append(str(path))

from transforms import MinMaxScaler


class TestScaler(unittest.TestCase):

    def setUp(self):
        self.Xs = []
        for i in range(10):
            self.Xs.append(numpy.random.normal(size=(10,10)))

        self.scaler = MinMaxScaler()

    def test_synthetic_data(self):
        for x in self.Xs:
            self.assertTrue(numpy.max(x) > 1 and numpy.min(x) < 0)

    def test_scaler(self):
        for x in self.Xs:
            self.scaler.partial_fit(x)
            x = self.scaler.transform(x)
            self.assertTrue(numpy.max(x) <= 1.0 and numpy.min(x) >= 0.0)

if __name__ == "__main__":
    unittest.main()
