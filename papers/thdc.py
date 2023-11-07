import torch
import numpy as np

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def hdv(d, dtype=torch.int8):
    v = torch.randint(0, 2, (d,), dtype=dtype, device="cuda")
    v[v == 0] = -1

    return v


def hdvs(d, n, dtype=torch.int8):
    vs = torch.randint(0, 2, (d, n), dtype=dtype, device="cuda")
    vs[vs == 0] = -1

    return vs


def hdvw(B, w):
    start = round(w * len(B))
    head = B[:start]
    tail = B[start:] * -1
    return torch.cat((head, tail), dim=0)


# a = hdv(10)
# print(a)
# print(hdvw(a, 0.5))


def bind(xs):
    return torch.prod(torch.stack(xs), 0)


# a = hdv(10)
# b = hdv(10)
# ba = torch.stack([a, b], 0)
# print(ba)


def bundle(xs):
    return torch.sum(torch.stack(xs), 0)


# print(bundle((a, b)))


def sbundle(xs):
    return torch.sign(torch.sum(torch.stack(xs), 0))


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


def convolution(vs, side=2, weight=20):
    size = len(vs) - 2 * side
    width = side * 2 + 1

    return [sbundle([*vs[i : i + width], weight * vs[i + side]]) for i in range(size)]


def hdvsc(n, d, side=2, weight=20, iter=5):
    vs = hdvs(n + iter * side * 2, d)
    for _ in range(iter):
        vs = convolution(vs, side, weight)

    return vs
