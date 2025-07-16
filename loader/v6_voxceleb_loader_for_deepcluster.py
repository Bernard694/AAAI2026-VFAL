import ipdb
import numpy as np
import torch
from utils import pickle_util, sample_util, worker_util, vec_util
from torch.utils.data import DataLoader

'''
创建数据加载器。
batch_size：批量大小。
full_length：数据集的总长度。
name2face_emb：面部路径到嵌入向量的字典。
name2voice_emb：语音路径到嵌入向量的字典。
movie2label：电影片段到伪标签的字典。
'''
def get_iter(batch_size, full_length, name2face_emb, name2voice_emb, movie2label_gender, movie2label_nation, movie2label_id):
    train_iter = DataLoader(DataSet(name2face_emb, name2voice_emb, full_length, movie2label_gender, movie2label_nation, movie2label_id),
                            batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter

class DataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb, full_length, movie2label_gender, movie2label_nation, movie2label_id):
        self.movie2label_gender = movie2label_gender
        self.movie2label_nation = movie2label_nation
        self.movie2label_id = movie2label_id

        self.train_movie_list = list(movie2label_id.keys())
        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb
        self.movie2label = movie2label_id

        # create movie2jpg, movie2wav dict
        self.movie2jpg_path = {}
        self.movie2wav_path = {}
        # 加载电影片段到图像和语音路径的映射
        name2movies = pickle_util.read_pickle("../dataset/info/name2movies.pkl")
        '''这个for循环要结合name2movies的数据结构来看'''
        for name, movie_list in name2movies.items():
            for movie_obj in movie_list:
                movie_name = movie_obj.replace("/1.6/", "/")
                # A.J._Buckley/J9lHsKG98U8
                filtered_keys = [k for k in name2face_emb.keys() if k.startswith(movie_name)]
                all_jpgs = []
                all_wavs = []
                for k in filtered_keys:
                    all_jpgs.append(k)
                    all_wavs.append(k)

                self.movie2wav_path[movie_name] = all_wavs
                self.movie2jpg_path[movie_name] = all_jpgs

    def __len__(self):
        return self.full_length

    '''作者写的：'''
    def __getitem__(self, index):
        '''# 随机选择一个电影片段'''
        # movie = sample_util.random_element(self.train_movie_list) #从伪标签里面去选择电影
        movie = self.train_movie_list[index % len(self.train_movie_list)]  # 根据索引循环取电影片段
        label = self.movie2label[movie] #伪标签

        '''# 随机选择该电影片段的一个图像和语音'''
        img = sample_util.random_element(self.movie2jpg_path[movie])
        wav = sample_util.random_element(self.movie2wav_path[movie])
        '''# 将路径转换为嵌入向量'''
        # wav, img = self.to_tensor(wav, img)

        '''# 加载原始嵌入数据'''
        emb_wav = self.name2voice_emb[wav]  # [array(192,), array(256,)]
        emb_face = self.name2face_emb[img]  # [array(512,), array(512,), array(512,), array(512,)]

        '''# 转换语音嵌入为tensor列表'''
        wav_tensors = [
            torch.as_tensor(emb_wav [0], dtype=torch.float32),  # (192,)
            torch.as_tensor(emb_wav [1], dtype=torch.float32)  # (256,)
        ]

        '''# 转换人脸嵌入为tensor列表'''
        face_tensors = [
            torch.as_tensor(arr, dtype=torch.float32)
            for arr in emb_face # 每个都是(512,)
        ]

        label_gender = self.movie2label_gender[movie]
        label_nation = self.movie2label_nation[movie]
        label_id = self.movie2label_id[movie]

        return (
            wav_tensors,
            face_tensors,
            torch.LongTensor([label_gender]),
            torch.LongTensor([label_nation]),
            torch.LongTensor([label_id])
        )

    def to_tensor(self, wav_path, img_path):
        ans = []

        emb_wav = self.name2voice_emb[wav_path] # 加载语音嵌入
        emb_wav = torch.FloatTensor(emb_wav)
        ans.append(emb_wav)

        emb_face = self.name2face_emb[img_path]
        emb_face = torch.FloatTensor(emb_face)
        ans.append(emb_face)

        return emb_wav,emb_face



