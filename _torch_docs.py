"""Adds docstrings to functions defined in the torch._C"""

import re

import torch._C

def parse_kwargs(desc):
    """Maps a description of args to a dictionary of {argname: description}.
    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # Split on exactly 4 spaces after a newline
    regx = re.compile("\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(' ')[0]: desc for desc in kwargs}


reduceops_common_args = parse_kwargs("""
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
""")

factory_common_args = parse_kwargs("""
    out (Tensor, optional): the output tensor
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
""")

factory_like_common_args = parse_kwargs("""
    input (Tensor): the size of :attr:`input` will determine size of the output tensor
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
""")

