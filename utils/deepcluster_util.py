import mkl
import collections
import random
import faiss
import numpy as np
import torch
from utils import wb_util, distance_util

mkl.get_max_threads()

def k2(X, k):
    n, d = 50, X.shape[1]
    faiss.normalize_L2(X)
    km = faiss.Kmeans(
        d, k, niter=n, verbose=True, spherical=True,
        min_points_per_centroid=3, max_points_per_centroid=1000000,
        gpu=True, nredo=10
    )
    km.train(X)
    D, I = km.index.search(X, 1)
    L = I.squeeze()
    sims = np.array([
        distance_util.cosine_similarity(X[i], km.centroids[L[i]])
        for i in range(len(X))
    ])
    return L, sims

def center(v, f, L, k):
    d = collections.defaultdict(list)
    for vi, fi, l in zip(v, f, L):
        d[l].extend([vi, fi])
    return np.array([np.mean(d[i], 0) for i in range(k)])

def cluster(keys, emb, v, f, k, t="all"):
    if t == "v":
        x = np.array(v)
    elif t == "f":
        x = np.array(f)
    elif t == "all":
        x = np.array(emb)
    else:
        raise ValueError
    L, sims = k2(x, k)
    m2l = {k: l for k, l in zip(keys, L)}
    if k == 1000:
        m2l = refine(m2l)
    c = center(v, f, L, k)
    return m2l, c

def refine(m2l):
    a2c = collections.defaultdict(lambda: collections.defaultdict(int))
    c2a = collections.defaultdict(lambda: collections.defaultdict(int))
    th = 0.7
    for k, v in m2l.items():
        a = k.split('/')[0]
        a2c[a][v] += 1
        c2a[v][a] += 1
    a2maj = {a: max(cnt, key=cnt.get) for a, cnt in a2c.items()}
    out = {}
    for k, v in m2l.items():
        a = k.split('/')[0]
        maj = a2maj[a]
        total = sum(a2c[a].values())
        conf = a2c[a].get(v, 0) / total
        if conf < th and len(c2a[v]) > 1:
            out[k] = maj
        else:
            out[k] = v
    return out