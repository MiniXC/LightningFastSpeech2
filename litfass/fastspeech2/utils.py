from time import time
import torch_xla.debug.metrics as met

class Timer:    
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        #self.start = time()
        # self.start_c = met.metric_data('CompileTime')[0]
        #self.start_t = met.metric_data('TransferFromServerTime')[0]
        #print(f"{self.name} start")
        return self

    def __exit__(self, *args):
        pass
        # self.end = time()
        # self.end_c = met.metric_data('CompileTime')[0]
        # self.end_t = met.metric_data('TransferFromServerTime')[0]
        # interval = self.end - self.start
        # interval_c = self.end_c - self.start_c
        # interval_t = self.end_t - self.start_t
        # print(f"{self.name} comp: {interval_c}")
        #print(f"{self.name} tran: {interval_t}")
        #print(f"{self.name} time: {interval:.03f}")
        # print(f"total comp: {self.end_c}")
        #print(f"total tran: {self.end_t}")
        #print()
        #print(f"{self.name} end (time: {interval:.03f})")


def bucketize(x, boundaries):
    return (x.unsqueeze(-1)>boundaries).sum(axis=-1)