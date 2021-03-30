

class Cycle:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = None

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.iterable)
        return self.iterator

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.iterable)
        try:
            obj = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            obj = next(self.iterator)
        return obj
