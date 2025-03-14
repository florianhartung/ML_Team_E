import torch
import sys
sys.path.append("/path/to/magicdl/parent")
from magicdl import magic

NUM_PIXELS = 1039

class HexaToParallelogram(torch.nn.Module):
    def __init__(self, padding_value=0.0):
        super(HexaToParallelogram, self).__init__()
        self._lt = magic.Geometry.lookup_table()
        self._padding_value = padding_value

    def forward(self, hexa):
        if len(hexa) > NUM_PIXELS:
            hexa = hexa[:NUM_PIXELS]
        qs, rs = (list(t) for t in zip(*self._lt.keys()))
        q_max, r_max = max(qs), max(rs)
        tensor = torch.full((2 * q_max + 1, 2 * r_max + 1), fill_value=self._padding_value)
        for q in range(2*q_max+1):
            for r in range(2*r_max+1):
                key = (q-q_max, r-r_max)
                if key in self._lt.keys():
                    tensor[q][r] = hexa[self._lt[key]]
        return tensor