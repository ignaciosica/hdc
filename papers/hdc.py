# Dependencies

import numpy as np
import functools as ft


# Basic arithemitic operations


def hdv(d):
    return np.random.choice([-1, 1], d)


def zero(d):
    return np.zeros((d,))


def bind(vs):
    return ft.reduce(lambda x, y: x * y, vs)


def bundle(vs):
    return ft.reduce(lambda x, y: x + y, vs)


def sbundle(vs):
    return np.sign(ft.reduce(lambda x, y: x + y, vs))


def pm(d):
    return np.random.shuffle(np.eye(d))


def inverse_pm(pm):
    return np.linalg.inv(pm)


def permute(pm, H):
    return pm.dot(H)


def cosine_similarity(A, B, norm_A=None, norm_B=None):
    dot_product = np.dot(A, B)

    if norm_A is None:
        norm_A = np.linalg.norm(A)

    if norm_B is None:
        norm_B = np.linalg.norm(B)

    if norm_A == 0 or norm_B == 0:
        return 0

    return dot_product / (norm_A * norm_B)


cosim = cosine_similarity


# Stochastic arithmetic


def weighted_averages(vs, ps):
    return np.fromiter(
        map(lambda t: np.random.choice([*t], p=ps), zip(*vs)), dtype=np.int_
    )


def hdva(B, a):
    return weighted_averages([B, -B], [(a + 1) / 2, (1 - a) / 2])


def hdvw(B, w):
    start = round(w * len(B))
    head = B[:start]
    tail = B[start:] * -1
    return np.concatenate([head, tail])


# Memory


class ItemMemory:
    def __init__(self):
        self.vectors = []

    def add_vector(self, label, V):
        self.vectors.append((label, V, np.linalg.norm(V)))

    def cleanup_aux(self, V):
        norm_V = np.linalg.norm(V)
        return max(self.vectors, key=lambda x: cosim(V, x[1], norm_V, x[2]))

    def cleanup(self, V):
        H = self.cleanup_aux(V)
        return (H[0], H[1], cosim(V, H[1]))