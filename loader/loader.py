import collections
import numpy as np
import torch
from utils import pickle_util, sample_util
from torch.utils.data import DataLoader

class DataSet(torch.utils.data.Dataset):
    def __init__(self, features, full_length, big_batch, batch_size, mode, labels=None):
        self.names = pickle_util.read_pickle("../dataset/info/train_valid_test_names.pkl")["train"]
        self.map = pickle_util.read_pickle("../dataset/info/name2tracks.pkl")
        self.batch_size = batch_size
        self.mode = mode
        if mode.startswith("sbc_"):
            self.ratio = float(mode.split("_")[-1])
            print("std_ratio", self.ratio)

        tracks = []
        for n in self.names:
            tracks += self.map[n]
        tracks = [t for t in tracks if t != "Moon_Bloodgood/1.6/JBhcgwl2pO0/2"]

        self.tracks = tracks
        self.full_length = full_length
        self.features = features
        self.big_batch_size = big_batch
        self.labels = labels
        self.actual_length = len(labels) if labels is not None else full_length

        self.labels_g = None
        self.labels_n = None
        self.labels_i = None

    def __len__(self):
        return self.actual_length

    def to_batch(self, tracks, arr_map):
        out = collections.defaultdict(list)
        assert len(tracks) == self.batch_size
        for t in tracks:
            arr = arr_map[t]
            for i in range(len(self.features)):
                out[i].append(arr[i])
        data = [np.array(out[i]) for i in range(len(self.features))]
        data = [torch.FloatTensor(d) for d in data]
        return data

    def __getitem__(self, idx):
        big_tracks, face, voice, arr_map = self.load_big()
        tracks = big_tracks[:self.batch_size]
        data = self.to_batch(tracks, arr_map)

        rate_before = self.rate(big_tracks)
        rate_after = self.rate(tracks)
        last_idx = big_tracks.tolist().index(tracks[-1])

        info = {
            "loader/before": rate_before,
            "loader/after": rate_after,
            "loader/desc": rate_before - rate_after,
            "loader/endIndex": last_idx,
        }

        g = [self.labels_g[t] for t in tracks]
        n = [self.labels_n[t] for t in tracks]
        i = [self.labels_i[t] for t in tracks]

        labels = [
            torch.tensor(g, dtype=torch.long).unsqueeze(1),
            torch.tensor(n, dtype=torch.long).unsqueeze(1),
            torch.tensor(i, dtype=torch.long).unsqueeze(1)
        ]

        return data, labels, info

    def load_one(self, track):
        path = "../dataset/features/" + track.replace("/1.6/", "/")
        d = pickle_util.read_pickle(path + "/compact.pkl")
        out = []
        for f in self.features:
            out.append(d[f])
        return out

    def load_big(self):
        big = sample_util.random_elements(self.tracks, self.big_batch_size)
        faces = []
        voices = []
        arr_map = {}
        for t in big:
            arr = self.load_one(t)
            arr_map[t] = arr
            faces.append(arr[0])
            voices.append(arr[-2])
        return big, faces, voices, arr_map

    def filter(self, tracks, faces, voices):
        f = np.array(faces)
        v = np.array(voices)
        fs = f @ f.T
        vs = v @ v.T
        cat = np.array([fs, vs])
        sim = np.min(cat, axis=0)
        np.fill_diagonal(sim, 0)
        mu = sim.mean()
        sigma = sim.std()
        thr = mu + self.ratio * sigma

        out = [tracks[0]]
        seen = {tracks[0].split("/")[0]}

        for i in range(1, len(tracks)):
            t = tracks[i]
            name = t.split("/")[0]
            if name in seen:
                sub = sim[i][:i]
                if sub.max() <= thr:
                    out.append(t)
                    seen.add(name)
                else:
                    sim[:, i] = 0
            else:
                out.append(t)
                seen.add(name)
            if len(out) == self.batch_size:
                break
        return out

    def rate(self, tracks):
        names = set([t.split("/")[0] for t in tracks])
        return 1 - len(names) / len(tracks)