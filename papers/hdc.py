# Dependencies

import numpy as np
import functools as ft


# Basic arithemitic operations


def hdv(d):
    return np.random.choice([-1, 1], d)


def zero(d):
    return np.zeros((d,))


def bind(xs):
    return ft.reduce(lambda x, y: x * y, xs)


def bundle(xs):
    return ft.reduce(lambda x, y: x + y, xs)


def bundleS(xs):
    return np.sign(ft.reduce(lambda x, y: x + y, xs))


def pm(d):
    return np.random.shuffle(np.eye(d))


def inversePm(P):
    return np.linalg.inv(P)


def permute(P, H):
    return P.dot(H)


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    if norm_A == 0 or norm_B == 0:
        return 0

    return dot_product / (norm_A * norm_B)


# Stochastic arithmetic


def weightedAverage(A, B, p, q):
    return np.fromiter(
        map(lambda t: np.random.choice([*t], p=[p, q]), zip(A, B)),
        dtype=np.int_,
    )


def weightedAverages(xs, ps):
    return np.fromiter(
        map(lambda t: np.random.choice([*t], p=ps), zip(xs)),
        dtype=np.int_,
    )


def hdvA(B, a):
    return weightedAverage(B, -B, (a + 1) / 2, (1 - a) / 2)


def hdvW(B, w):
    start = round(w * len(B))
    head = B[:start]
    tail = B[start:] * -1
    return np.concatenate([head, tail])


# Memory


class ItemMemory:
    def __init__(self, vectors=[]):
        self.vectors = vectors

    def addVector(self, label, V):
        self.vectors.append((label, V))

    def cleanup(self, V):
        return max(self.vectors, key=lambda x: cosine_similarity(V, x[1]))
