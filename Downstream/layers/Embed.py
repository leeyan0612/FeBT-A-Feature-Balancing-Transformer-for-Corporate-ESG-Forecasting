import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        if freq == 'year':
            self.year_embed = Embed(1, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:, :, 3])
        # weekday_x = self.weekday_embed(x[:, :, 2])
        # day_x = self.day_embed(x[:, :, 1])
        # month_x = self.month_embed(x[:, :, 0])
        year_x = self.year_embed(x)

        # return hour_x + weekday_x + day_x + month_x + minute_x
        return year_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, 'y': 1}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class AdapterLayer(nn.Module):
    def __init__(self, size, down_sample, up_sample):
        super(AdapterLayer, self).__init__()
        self.down_sample = nn.Linear(size, down_sample)
        self.up_sample = nn.Linear(down_sample, up_sample)

    def forward(self, x):
        down_sampled = self.down_sample(x)
        activated = nn.ReLU()(down_sampled)
        up_sampled = self.up_sample(activated)
        output = x + up_sampled
        return output


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        # self.vae_fc1 = nn.Linear(128, 128)
        # self.adapter1 = AdapterLayer(8, 4, 8)
        # self.adapter2 = AdapterLayer(8, 4, 8)
        # self.adapter3 = AdapterLayer(8, 4, 8)
        # self.adapter4 = AdapterLayer(8, 4, 8)

    def forward(self, x, x_mark):
        # x_1 = x[:, :, 31:63]
        # x_1[:, :, 0:8] = self.adapter1(x_1[:, :, 0:8].clone())
        # x_1[:, :, 8:16] = self.adapter1(x_1[:, :, 8:16].clone())
        # x_1[:, :, 16:24] = self.adapter1(x_1[:, :, 16:24].clone())
        # x_1[:, :, 24:32] = self.adapter1(x_1[:, :, 24:32].clone())
        # x_2 = x[:, :, 63:95]
        # x_2[:, :, 0:8] = self.adapter2(x_2[:, :, 0:8].clone())
        # x_2[:, :, 8:16] = self.adapter2(x_2[:, :, 8:16].clone())
        # x_2[:, :, 16:24] = self.adapter2(x_2[:, :, 16:24].clone())
        # x_2[:, :, 24:32] = self.adapter2(x_2[:, :, 24:32].clone())
        # x_3 = x[:, :, 95:127]
        # x_3[:, :, 0:8] = self.adapter3(x_3[:, :, 0:8].clone())
        # x_3[:, :, 8:16] = self.adapter3(x_3[:, :, 8:16].clone())
        # x_3[:, :, 16:24] = self.adapter3(x_3[:, :, 16:24].clone())
        # x_3[:, :, 24:32] = self.adapter3(x_3[:, :, 24:32].clone())
        # x_4 = x[:, :, 127:159]
        # x_4[:, :, 0:8] = self.adapter4(x_4[:, :, 0:8].clone())
        # x_4[:, :, 8:16] = self.adapter4(x_4[:, :, 8:16].clone())
        # x_4[:, :, 16:24] = self.adapter4(x_4[:, :, 16:24].clone())
        # x_4[:, :, 24:32] = self.adapter4(x_4[:, :, 24:32].clone())
        # # x_ = self.vae_fc1(x_.clone())  # 使用clone()创建副本
        # x[:, :, 31:63] = x_1
        # x[:, :, 63:95] = x_2
        # x[:, :, 95:127] = x_3
        # x[:, :, 127:159] = x_4
        #
        # x_ = x[:, :, 31:159]
        # x_ = self.vae_fc1(x_.clone())  # 使用clone()创建副本
        # x[:, :, 31:159] = x_
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding1(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, h_dim=16):
        super(DataEmbedding1, self).__init__()
        self.h_dim = h_dim
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.vae_fc1 = nn.Linear(4 * h_dim, 4 * h_dim)
        self.adapter1 = AdapterLayer(8, 4, 8)
        self.adapter2 = AdapterLayer(8, 4, 8)
        self.adapter3 = AdapterLayer(8, 4, 8)
        self.adapter4 = AdapterLayer(8, 4, 8)

    def forward(self, x, x_mark):
        x_1 = x[:, :, 31:31+self.h_dim]
        for i in range(0, self.h_dim, 8):
            x_1[:, :, i:i + 8] = self.adapter1(x_1[:, :, i:i + 8].clone())
        x_2 = x[:, :, 31+self.h_dim:31+2*self.h_dim]
        for i in range(0, self.h_dim, 8):
            x_2[:, :, i:i + 8] = self.adapter1(x_1[:, :, i:i + 8].clone())
        x_3 = x[:, :, 31+2*self.h_dim:31+3*self.h_dim]
        for i in range(0, self.h_dim, 8):
            x_3[:, :, i:i + 8] = self.adapter1(x_1[:, :, i:i + 8].clone())
        x_4 = x[:, :, 31+3*self.h_dim:31+4*self.h_dim]
        for i in range(0, self.h_dim, 8):
            x_4[:, :, i:i + 8] = self.adapter1(x_1[:, :, i:i + 8].clone())
        x[:, :, 31:31+self.h_dim] = x_1
        x[:, :, 31+self.h_dim:31+2*self.h_dim] = x_2
        x[:, :, 31+2*self.h_dim:31+3*self.h_dim] = x_3
        x[:, :, 31+3*self.h_dim:31+4*self.h_dim] = x_4
        x_ = x[:, :, 31:31+4*self.h_dim]
        x_ = self.vae_fc1(x_.clone())  # 使用clone()创建副本
        x[:, :, 31:31+4*self.h_dim] = x_
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
