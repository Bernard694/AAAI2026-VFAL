import ipdb
import numpy as np
import torch
from utils import pickle_util, worker_util
from utils.path_util import look_up
from torch.utils.data import DataLoader
from utils.config_load import face_emb_dict, voice_emb_dict
import collections


def extract(f_map, v_map, m):
    f_iter = build_iter(512, f_map, v_map, True)
    m1, e1 = core(f_iter, m, True)

    v_iter = build_iter(512, f_map, v_map, False)
    m2, e2 = core(v_iter, m, False)

    assert len(m1) == len(m2)
    out = np.hstack([e2, e1])
    return m1, out, e2, e1


def core(loader, net, flag):
    net.eval()
    dev = next(net.parameters()).device
    d = collections.defaultdict(list)
    for batch in loader:
        with torch.no_grad():
            names, xs = batch
            xs = [x.to(dev) for x in xs]
            if flag:
                emb = net.face_encoder(xs)
            else:
                xs = [x.float() for x in xs]
                emb = net.voice_encoder(xs)
            for e, n in zip(emb, names):
                d[n].append(e.detach().cpu().numpy())
    net.train()

    res = {}
    for k, v in d.items():
        res[k] = np.mean(v, axis=0)

    keys = sorted(res.keys())
    mat = np.array([res[k] for k in keys])
    return keys, mat


def build_iter(bs, f_map, v_map, flag):
    ds = OrdSet(flag, f_map, v_map)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      pin_memory=True, worker_init_fn=worker_util.worker_init_fn)


class OrdSet(torch.utils.data.Dataset):
    def __init__(self, flag, f_map, v_map):
        nm = pickle_util.read_pickle("../dataset/info/name2movies.pkl")
        tn = pickle_util.read_pickle("../dataset/info/train_valid_test_names.pkl")["train"]

        mvs = []
        jpgs = []
        wavs = []

        for n in tn:
            for m in nm[n]:
                m = m.replace("/1.6/", "/")
                mvs.append(m)
                for k in f_map:
                    if k.startswith(m):
                        if flag:
                            jpgs.append([m, k])
                        else:
                            wavs.append([m, k])

        if flag:
            self.data = jpgs
            self.emap = f_map
        else:
            self.data = wavs
            self.emap = v_map
        self.flag = flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        m, p = self.data[idx]
        x = self.emap[p]
        return m, x