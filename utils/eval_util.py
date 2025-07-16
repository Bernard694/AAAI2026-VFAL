import ipdb
import torch
from torch.utils.data import DataLoader
import time

from utils import pickle_util, arr_util, vec_util, cuda_util
from utils.eval_core_func import *
from utils.path_util import look_up


class DataSet(torch.utils.data.Dataset):

    def __init__(self, data, features):
        self.data = data
        self.features = features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        track = self.data[index]
        if ".wav" in track:
            # Luke_Arnold/XTs3NSW63dY/00001.wav
            tmp = track.split("/")
            clip = int(tmp[-1].replace(".wav", ""))
            track = "../dataset/features/%s/%s/%s/" % (tmp[0], tmp[1], clip)
        else:
            track = "../dataset/features/" + track.replace("/1.6/", "/")

        tmp_dict = pickle_util.read_pickle(look_up(track + "/compact.pkl"))

        all_features = []
        for f in self.features:
            data = torch.FloatTensor(tmp_dict[f])
            all_features.append(data)

        return all_features, self.data[index]

'''生成路径到嵌入的字典，用于评估模型性能。'''
def get_track2emb(encoder, path_set, features):
    loader = DataLoader(DataSet(list(path_set), features), batch_size=512, shuffle=False, pin_memory=True) #创建一个 DataLoader，用于批量加载数据。
    with torch.no_grad():
        t1 = time.time()
        p2emb = {}
        '''遍历数据加载器，生成每个路径对应的嵌入。'''
        for data, key_list in loader:
            data = cuda_util.to_cuda(data)
            emb_batch = encoder(data).detach().cpu().numpy()
            for key, emb in zip(key_list, emb_batch):
                p2emb[key] = emb
        t2 = time.time()
    return p2emb


class EmbEva:
    '''初始化评估类，分离面部和语音特征。'''
    def __init__(self, features):
        self.face_features = [f for f in features if f.startswith("f_")]
        self.voice_features = [f for f in features if f.startswith("v_")]

    '''执行验证评估，返回验证集上的 AUC 值'''
    def do_valid(self, model):
        obj = {
            "valid/auc": self.do_verification(model, "../dataset/evals/valid_verification.pkl")
        }
        return obj

    '''执行测试集（GN）上的验证评估。'''
    def do_test_gn(self, model):
        return self.do_verification(model, "../dataset/evals/test_verification_gn.pkl")

    '''执行全面的测试评估，包括验证、检索和匹配任务'''
    def do_test(self, model):
        obj = {}
        obj["test/auc"] = self.do_verification(model, "../dataset/evals/test_verification.pkl")
        obj["test/auc_g"] = self.do_verification(model, "../dataset/evals/test_verification_g.pkl")
        obj["test/auc_n"] = self.do_verification(model, "../dataset/evals/test_verification_n.pkl")

        # 2.retrieval
        obj["test/map_v2f"], obj["test/map_f2v"] = self.do_retrival(model, "../dataset/evals/test_retrieval.pkl")

        # 3.matching
        obj["test/ms_v2f"], obj["test/ms_f2v"] = self.do_matching(model, "../dataset/evals/test_matching.pkl")
        obj["test/ms_v2f_g"], obj["test/ms_f2v_g"] = self.do_matching(model, "../dataset/evals/test_matching_g.pkl")
        return obj


    '''读取验证数据，生成嵌入字典，并计算验证指标。'''
    def do_verification(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        return calc_vrification(data["list"], v2emb, f2emb)

    '''读取匹配数据，生成嵌入字典，并计算匹配指标。'''
    def do_matching(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        ms_vf, ms_fv = calc_ms(data["match_list"], v2emb, f2emb)
        return ms_vf, ms_fv

    '''读取检索数据，生成嵌入字典，并计算检索指标。'''
    def do_retrival(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        map_vf, map_fv = calc_map_value(data["retrieval_lists"], v2emb, f2emb)
        return map_vf, map_fv

    '''执行 1:N 匹配任务，计算匹配结果。'''
    # def do_1_N_matching(self, model):
    #     data = pickle_util.read_pickle(path_util.look_up("../dataset/evals/test_matching_1N.pkl"))
    #     # 1.生成向量表示
    #     v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
    #     key2emb = {**v2emb, **f2emb}
    #     ans = {}
    #     ans["v2f"] = handle_1_n(data["match_list"], is_v2f=True, key2emb=key2emb)
    #     ans["f2v"] = handle_1_n(data["match_list"], is_v2f=False, key2emb=key2emb)
    #     return ans

    def do_full_test_v2(self, model, gender_constraint=False):
        obj = {}
        # 1.verification
        t1 = time.time()
        obj["test/auc"] = self.do_verification(model, "../dataset/evals/test_verification.pkl")
        obj["test/auc_g"] = self.do_verification(model, "../dataset/evals/test_verification_g.pkl")
        obj["test/auc_n"] = self.do_verification(model, "../dataset/evals/test_verification_n.pkl")
        t2 = time.time()

        # 2.retrieval
        obj["test/map_v2f"], obj["test/map_f2v"] = self.do_retrival(model, "../dataset/evals/test_retrieval.pkl")
        t3 = time.time()

        # 3.matching
        obj["test/ms_v2f"], obj["test/ms_f2v"] = self.do_matching(model, "../dataset/evals/test_matching.pkl")
        obj["test/ms_v2f_g"], obj["test/ms_f2v_g"] = self.do_matching(model, "../dataset/evals/test_matching_g.pkl")
        t4 = time.time()

        # 4. 1-N matching
        t5 = time.time()
        matching_1n_result = self.do_1_N_matching(model)
        # 给所有键添加test/前缀
        prefixed_obj = {f"test/{k}": v for k, v in matching_1n_result.items()}
        obj.update(prefixed_obj)
        t6 = time.time()

        print("time spend: auc%.1f map%.1f matching%.1f 1n_matching%.1f" % (t2 - t1, t3 - t2, t4 - t3, t6 - t5))
        return obj


    '''执行 1-N 匹配任务，返回匹配准确率。'''

    def do_1_N_matching(self, model):
        # data = pickle_util.read_pickle(path_util.look_up("../dataset/evals/test_matching_10.pkl"))
        data = pickle_util.read_pickle("../dataset/evals/test_matching_1N.pkl")
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        key2emb = {**v2emb, **f2emb}  # 合并语音和面部的嵌入字典，方便后续统一查询
        # ans = {}
        v2f_result = handle_1_n(data["match_list"], is_v2f=True, key2emb=key2emb)
        v2f_prefixed = {f"v2f_{k}": v for k, v in v2f_result.items()}

        f2v_result = handle_1_n(data["match_list"], is_v2f=False, key2emb=key2emb)
        f2v_prefixed = {f"f2v_{k}": v for k, v in f2v_result.items()}

        return {**v2f_prefixed, **f2v_prefixed}

    '''将模型设置为评估模式，生成面部和语音特征的嵌入字典，然后恢复训练模式。'''
    def to_emb_dict(self, model, jpg_set, wav_set):
        model.eval()
        f2emb = get_track2emb(model.face_encoder, jpg_set, self.face_features)
        v2emb = get_track2emb(model.voice_encoder, wav_set, self.voice_features)
        model.train()
        return v2emb, f2emb
