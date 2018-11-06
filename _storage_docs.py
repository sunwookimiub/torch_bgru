"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = [
    'DoubleStorageBase',
    'FloatStorageBase',
    'LongStorageBase',
    'IntStorageBase',
    'ShortStorageBase',
    'CharStorageBase',
    'ByteStorageBase',
]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass

