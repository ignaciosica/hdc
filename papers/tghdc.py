import hdc
from tinygrad.tensor import Tensor
import tinygrad.mlops as mlops
from tinygrad.helpers import prod

import numpy as np


def hdv(d):
    return Tensor(hdc.hdv(d))


def zero(d):
    return Tensor(hdc.zero(d))


def bind(vs):
    return prod(vs)


def bundle(vs):
    return Tensor.stack(vs).sum(axis=0)


# TODO
def sbundle(vs):
    return Tensor.stack(vs).sum(axis=0)


def cosine_similarity(A, B, norm_A=None, norm_B=None):
    dot_product = A.dot(B)

    if norm_A is None:
        norm_A = np.linalg.norm(A.numpy())

    if norm_B is None:
        norm_B = np.linalg.norm(B.numpy())

    if norm_A == 0 or norm_B == 0:
        return 0

    return dot_product / (norm_A * norm_B)


a = hdv(10000)
b = hdv(10000)
c = hdv(10000)

s = Tensor.stack([a, b, c])

p = prod([a, b, c])
print("p ", p.numpy())
bi = bind([a, b, c])
print("bi", bi.numpy())
# sumt = s._reduce(mlops.Sum, axis=0)
# t = s.sum(axis=0)


print(s.numpy())
# print(sumt.numpy())
# print(t.numpy())
# print(bundle([a, b, c]).numpy())
print(sbundle([a, b, c]).numpy())

print(cosine_similarity(a, b).numpy())
