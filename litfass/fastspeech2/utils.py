from time import time

class Timer:    
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        interval = self.end - self.start
        print(f'{self.name} took {interval:.03f} sec.')

def bucketize(x, boundaries):
    return (x.unsqueeze(-1)>boundaries).sum(axis=-1)
