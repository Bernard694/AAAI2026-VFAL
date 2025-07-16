import ipdb
import numpy as np
import torch
from utils import pickle_util, sample_util, worker_util, vec_util
from torch.utils.data import DataLoader


def get_iter(bs, l, f_map, v_map, g, n, i):
    ds = DataSet(f_map, v_map, l, g, n, i)
    return DataLoader(ds, batch_size=bs, shuffle=True,
                      pin_memory=True, worker_init_fn=worker_util.worker_init_fn)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, f_map, v_map, l, g, n, i):
        self.g = g
        self.n = n
        self.i = i
        self.keys = list(i.keys())
        self.l = l
        self.f_map = f_map
        self.v_map = v_map

        self.m2j = {}
        self.m2w = {}
        nm = pickle_util.read_pickle("../dataset/info/name2movies.pkl")
        for name, ms in nm.items():
            for m in ms:
                m = m.replace("/1.6/", "/")
                ks = [k for k in f_map if k.startswith(m)]
                self.m2j[m] = ks
                self.m2w[m] = ks

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        m = self.keys[idx % len(self.keys)]
        img = sample_util.random_element(self.m2j[m])
        wav = sample_util.random_element(self.m2w[m])

        ev = self.v_map[wav]
        ef = self.f_map[img]

        wt = [torch.as_tensor(ev[0], dtype=torch.float32),
              torch.as_tensor(ev[1], dtype=torch.float32)]
        ft = [torch.as_tensor(e, dtype=torch.float32) for e in ef]

        return wt, ft, torch.LongTensor([self.g[m]]), torch.LongTensor([self.n[m]]), torch.LongTensor([self.i[m]])

    def to_tensor(self, wp, ip):
        ev = torch.FloatTensor(self.v_map[wp])
        ef = torch.FloatTensor(self.f_map[ip])
        return ev, ef