class MeanCalculater:
    def __init__(self):
        self.value = 0.
        self.i = 0

    def clear(self):
        self.value = 0.
        self.i = 0

    def add_value(self, value):
        self.value += value
        self.i += 1

    @property
    def mean_value(self):
        return 0 if self.i == 0 else self.value / self.i