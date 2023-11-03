import torch
import numpy as np

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def hdv(d):
    return torch.sign(torch.randint(-100000, 100000, (d,), dtype=torch.float))


def bind(xs):
    return torch.prod(torch.cat(xs), axis=0)


def bundle(xs):
    return torch.sum(torch.cat(xs), axis=0)


def sbundle(xs):
    return torch.sign(torch.sum(torch.cat(xs), axis=0))


def cosine_similarity(A, B, norm_A=None, norm_B=None):
    dot_product = torch.dot(A, B)

    if norm_A is None:
        norm_A = torch.norm(A)

    if norm_B is None:
        norm_B = torch.norm(B)

    if norm_A == 0 or norm_B == 0:
        return 0

    return torch.div(dot_product, norm_A * norm_B)


cosim = cosine_similarity


class ItemMemory:
    def __init__(self):
        self.vectors = []

    def add_vector(self, label, V):
        self.vectors.append((label, V, torch.norm(V)))

    def add_vector_wn(self, label, V):
        self.vectors.append((label, V))

    def cleanup_aux(self, V):
        norm_V = torch.norm(V)
        return max(self.vectors, key=lambda x: cosim(V, x[1], norm_V, x[2]))

    def cleanup(self, V):
        H = self.cleanup_aux(V)
        return (H[0], H[1], cosim(V, H[1]))

    def cleanup_all_aux(self, V, n=10):
        norm_V = torch.norm(V)
        return sorted(self.vectors, key=lambda x: cosim(V, x[1], norm_V), reverse=True)[
            :n
        ]

    def cleanup_all(self, V, n=10):
        return [(H[0], H[1], cosim(V, H[1])) for H in self.cleanup_all_aux(V, n=n)]


def hdvs(n, d):
    return [hdv(d) for _ in range(n)]


def convolution(vs, side=2, weight=20):
    size = len(vs) - 2 * side
    width = side * 2 + 1

    return [sbundle([*vs[i : i + width], weight * vs[i + side]]) for i in range(size)]


def hdvsc(n, d, side=2, weight=20, iter=5):
    vs = hdvs(n + iter * side * 2, d)
    for _ in range(iter):
        vs = convolution(vs, side, weight)

    return vs
