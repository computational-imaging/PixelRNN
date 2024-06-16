from torch.nn import Parameter

import torch


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    def forward(input, *args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w)
            # setattr(module, name_w, raw_w)
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)

        return torch.nn.functional.linear(input, w, bias=getattr(module, 'bias'))

    setattr(module, 'forward', forward)


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)