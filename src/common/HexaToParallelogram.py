import torch
import torch.nn as nn
from magicdl import magic
from src.common import NUM_PIXELS

class HexaToParallelogram(nn.Module):
    def __init__(self, pixel_dim, padding_value=0.0):
        """
        param pixel_dim: Along which dimension the 1039 pixels are placed.
        """
        super().__init__()
        self._lt = magic.Geometry.lookup_table()
        self._padding_value = padding_value
        self._pixel_dim = pixel_dim

    def forward(self, hexa):
        hexa = hexa.transpose(self._pixel_dim, -1)

        batch_shape = hexa.shape[:-1]
        num_pixels = hexa.shape[-1]
        
        if num_pixels > NUM_PIXELS:
            hexa = hexa[..., :NUM_PIXELS]

        qs, rs = (list(t) for t in zip(*self._lt.keys()))
        q_max, r_max = max(qs), max(rs)

        tensor_shape = batch_shape + (2 * q_max + 1, 2 * r_max + 1)
        tensor = torch.full(tensor_shape, fill_value=self._padding_value)

        if hexa.device.type != 'cpu':
            tensor = tensor.to(hexa.device)

        for q in range(2*q_max+1):
            for r in range(2*r_max+1):
                key = (q-q_max, r-r_max)
                if key in self._lt.keys():
                    tensor[..., q, r] = hexa[..., self._lt[key]]
        return tensor