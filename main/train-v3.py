import os
import torch
import numpy as np
from utils import myparser, seed_util, wb_util, model_util, pickle_util, eval_util, model_selector, unsup_nce, cuda_util, worker_util, deepcluster_util
from models import transformer
from loaders import v5_voxceleb_cluster_ordered_loader, v6_voxceleb_loader_for_deepcluster
from utils.config_load import face_emb_dict, voice_emb_dict
from pytorch_metric_learning import losses, miners

def step(ep, st, batch):
    opt.zero_grad()
    v, f, g, n, i = batch
    v = [t.cuda() for t in v]
    f = [t.cuda() for t in f]
    x = f + v
    ve, fe = net(x)
    e = torch.cat([ve, fe], 0)
    labs = [torch.cat([t, t], 0).squeeze() for t in [g, n, i]]
    l_g = loss_ms(e, labs[0])
    l_n = loss_ms(e, labs[1])
    l_id = loss_ms(e, labs[2])
    trips = miner(e, labs[2])
    l_trip = loss_trip(e, labs[2], trips)
    loss = l_g * args.r0 + args.r1 * l_n + l_id * args.r2 + l_trip * args.r3
    loss.backward()
    opt.step()
    return loss.item(), {
        "l_g": l_g.item(),
        "l_n": l_n.item(),
        "l_id": l_id.item(),
        "l_t": l_trip.item()
    }

def train():
    st = 0
    net.train()
    best = {}
    es = 0
    for ep in range(args.epoch):
        wb_util.log({"ep": ep})
        keys, emb, ev, ef = v5_voxceleb_cluster_ordered_loader.extract(face_emb_dict, voice_emb_dict, net)
        m2g, _ = deepcluster_util.do_cluster_v2(keys, emb, ev, ef, 2, "all")
        m2n, _ = deepcluster_util.do_cluster_v2(keys, emb, ev, ef, 32, "all")
        m2i, _ = deepcluster_util.do_cluster_v2(keys, emb, ev, ef, 1000, "all")
        itr = v6_voxceleb_loader_for_deepcluster.get_iter(args.bs, args.bpe * args.bs,
                                                          face_emb_dict, voice_emb_dict,
                                                          m2g, m2n, m2i)
        for batch in itr:
            loss, info = step(ep, st, batch)
            st += 1
            if st % 50 == 0:
                obj = {"st": st, "loss": loss, **info}
                print(obj)
                wb_util.log(obj)
            if st % args.eval == 0:
                net.eval()
                res = eva(net)
                sel.log(res)
                ind = "valid/auc"
                if res[ind] < 80:
                    obj = res
                    print("skip test")
                else:
                    tst = eva.test(net)
                    obj = {**res, **tst}
                    model_util.rm_last()
                    name = "auc[%.2f,%.2f]_ms[%.2f,%.2f,%.2f,%.2f]_map[%.2f,%.2f].pkl" % (
                        tst["test/auc"], tst["test/auc_g"],
                        tst["test/ms_v2f"], tst["test/ms_f2v"],
                        tst["test/ms_v2f_g"], tst["test/ms_f2v_g"],
                        tst["test/map_v2f"], tst["test/map_f2v"])
                    path = os.path.join(args.save, args.proj, args.name, name)
                    model_util.save(0, net, None, path)
                    pickle_util.save_json(path + ".json", tst)
                    if best is None:
                        best = tst
                    else:
                        es += 1
                        print("es:", es)
                        for k, v in tst.items():
                            if k not in best or v > best[k]:
                                es = 0
                                best[k] = v
                wb_util.log(obj)
                print(obj)
                wb_util.init(args)
                net.train()
                if es > 10:
                    print("early stop")
                    wb_util.log({f"best_{k}": v for k, v in best.items()})
                    return

if __name__ == "__main__":
    p = myparser.MyParser(epoch=200, bs=256, lr=2e-4, save="outputs", es=10, w=4)
    p.custom({
        "load": "",
        "bpe": 500,
        "eval": 50,
        "fts": "f_2plus1D_512,f_swin_512,f_mobile_512,f_dynamic_incept_512",
        "vts": "v_ecapa_192,v_resemble_256",
        "pos": False,
        "std": -1.0,
        "din": 128,
        "head": 8,
        "ff": 2048,
        "layer": 1,
        "drop": 0.3,
        "proj": False,
        "big": 512,
        "mode": "sbc_4.0",
        "temp": 0.07,
        "nc": 901,
        "ci": 5,
        "num": 901,
        "alpha": 2.0,
        "beta": 50.0,
        "base": 1.0,
        "ctype": "all",
        "mse": 0.0,
        "r0": 0.1,
        "r1": 0.1,
        "r2": 0.8,
        "r3": 0.0
    })
    p.use_wb("EFT-MIX-2", "no-triplet")
    args = p.parse()
    seed_util.set_seed(args.seed)

    fs = args.fts.split(",") + args.vts.split(",")
    net = transformer.Model(fs, args.din, args.head, args.ff, args.layer,
                            args.drop, args.std, args.proj, args.pos)
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_ms = losses.MultiSimilarityLoss(alpha=args.alpha, beta=args.beta, base=args.base)
    loss_trip = losses.TripletMarginLoss(margin=0.2, swap=True)
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
    eva = eval_util.EmbEva(fs)
    sel = model_selector.ModelSelector()
    train()