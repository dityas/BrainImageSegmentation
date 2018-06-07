import numpy


class MinMaxScaler:

    """
        Scales tensors between 0 and 1.
    """

    def __init__(self):
        self.prev_max = 0.0
        self.prev_min = 0.0
        pass

    def partial_fit(self, X, y=None):
        """
            Updates observed max and min values.
        """

        curr_max = numpy.max(X)
        curr_min = numpy.min(X)

        if curr_max > self.prev_max:
            self.prev_max = curr_max
        if curr_min < self.prev_min:
            self.prev_min = curr_min

    def transform(self, X, y=None):
        """
            Does online scaling based on min and max values observed so far.
        """
        X = (X - self.prev_min) / (self.prev_max - self.prev_min + 0.0000001)
        return X
