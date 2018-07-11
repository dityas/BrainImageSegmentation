
class EarlyStopper:
    """
        Implements early stopping with patience.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.violations = 0
        self.min_val = float("inf")

    def check_stop_training(self, metric):

        if metric < self.min_val:
            self.violations = 0
            self.min_val = metric
            return False

        elif self.violations == self.patience:
            return True

        else:
            self.violations += 1
            return False
