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
    return ft.reduce(lambda x, y: np.add(x, y), vs)


def sbundle(vs):
    return np.sign(ft.reduce(lambda x, y: x + y, vs))


def pm(d):
    P = np.eye(d)
    np.random.shuffle(P)
    return P


def inverse_pm(pm):
    return np.linalg.inv(pm)


def permute(pm, v):
    return pm.dot(v)


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


# B = hdv(10000)
# a = 0.7
# b = 0.3
# Ba = hdva(B, a)
# Bb = hdva(B, b)

# A = hdv(10000)
# C = hdv(10000)

# Ta = weighted_averages([A, C], [0.64, 0.36])
# Tb = weighted_averages([A, C], [0.64, 0.36])
# print("A  - Ta", cosim(A, Ta))
# print("A  - Tb", cosim(A, Tb))
# print("B  - Ta", cosim(C, Ta))
# print("B  - Tb", cosim(C, Tb))

# print("Ta - Tb", cosim(Ta, Tb))

# Taa = bundle([A * 10, C * 1])
# Tbb = bundle([A * 10, C * 1])
# print("A  - Taa", cosim(A, Taa))
# print("A  - Tbb", cosim(A, Tbb))
# print("C  - Taa", cosim(C, Taa))
# print("C  - Tbb", cosim(C, Tbb))
# print("Taa  - Tbb", cosim(Taa, Tbb))
# print(cosim(B, Ba))
# print(cosim(B, Bb))
# print(cosim(Ba, Bb))


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


# Linear convolution


def hdvs(n, d):
    return [hdv(d) for _ in range(n)]


def convolution(vs, side=2, weight=20):
    size = len(vs) - 2 * side
    width = side * 2 + 1

    return np.array(
        [sbundle([*vs[i : i + width], weight * vs[i + side]]) for i in range(size)]
    )


def hdvsc(n, d, side=2, weight=20, iter=5):
    vs = hdvs(n + iter * side * 2, d)
    for _ in range(iter):
        vs = convolution(vs, side, weight)

    return vs
