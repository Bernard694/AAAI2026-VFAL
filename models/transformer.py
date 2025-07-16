import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.position_encode import PositionalEncoding
import torch.nn.functional as F

class GatedMoE(nn.Module):
    def __init__(self, d_in, d_h, n_e):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(d_in * n_e, d_h),
            nn.ReLU(),
            nn.Linear(d_h, n_e),
            nn.Softmax(dim=-1)
        )

    def forward(self, xs):
        xs = torch.stack(xs, 1)
        B, N, D = xs.shape
        w = self.g(xs.view(B, N * D)).unsqueeze(-1)
        out = torch.sum(w * xs, 1)
        return out

class Model(nn.Module):
    def __init__(self, fs, d, h, ff, l, drop, std, proj, pos, n_cls=900):
        super().__init__()
        self.d = d
        self.proj = proj
        enc = TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.enc = TransformerEncoder(enc, l)

        self.use_tok = std > 0
        if self.use_tok:
            def init(m):
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.data.normal_(0.0, std)
            self.tok = nn.Embedding(2, d)
            self.tok.apply(init)

        self.use_pos = pos
        if pos:
            self.pos = PositionalEncoding(d)
            self.r = 1e-5

        vs = [f for f in fs if f.startswith("v")]
        fs_ = [f for f in fs if f.startswith("f")]
        self.nf = len(fs_)
        self.make_proj("v", vs)
        self.make_proj("f", fs_)
        d_out = 128
        self.fp = self.mk_out(d, d_out)
        self.vp = self.mk_out(d, d_out)
        self.vmoe = GatedMoE(d, 64, len(vs))
        self.fmoe = GatedMoE(d, 64, len(fs_))

    def mk_out(self, i, o):
        return nn.Sequential(nn.Linear(i, o)) if self.proj else nn.Identity()

    def make_proj(self, tag, lst):
        for i, n in enumerate(lst):
            d_raw = int(n.split("_")[-1])
            setattr(self, f"proj_{tag}{i}", nn.Sequential(nn.Dropout(), nn.Linear(d_raw, self.d)))

    def forward(self, xs):
        nf = self.nf
        f = xs[:nf]
        v = xs[nf:]
        ve = self.v_enc(v)
        fe = self.f_enc(f)
        if ve.dim() == 1:
            ve = ve.unsqueeze(0)
        if fe.dim() == 1:
            fe = fe.unsqueeze(0)
        return ve, fe

    def f_enc(self, xs):
        return self.fp(self.encode(xs, "f"))

    def v_enc(self, xs):
        return self.vp(self.encode(xs, "v"))

    def encode(self, xs, tag):
        assert tag in ["v", "f"]
        fs = [getattr(self, f"proj_{tag}{i}")(xs[i]) for i in range(len(xs))]
        fused = self.fmoe(fs) if tag == "f" else self.vmoe(fs)
        if self.use_tok:
            idx = ["v", "f"].index(tag)
            b = xs[0].shape[0]
            fused = fused + self.tok(torch.tensor([idx] * b, device=fused.device))
        return self.out(fused.unsqueeze(1))

    def out(self, x):
        if self.use_pos:
            x = self.pos(x, self.r)
        x = self.enc(x).mean(1)
        return x