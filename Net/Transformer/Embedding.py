import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化TokenEmbedding类。

        参数:
        - vocab_size (int): 词汇表的大小。
        - d_model (int): 嵌入向量的维度。

        初始化父类nn.Embedding，并设置padding_idx为1。
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device):
        """
        初始化PositionalEmbedding类。

        参数:
        - d_model (int): 嵌入向量的维度。
        - max_len (int): 序列的最大长度。
        - device: 模型运行的设备。

        初始化位置编码矩阵，并计算位置编码。
        """
        super(PositionalEmbedding, self).__init__()
        # 创建一个形状为(max_len, d_model)的零张量，用于存储位置编码
        self.encoding = torch.zeros(max_len, d_model, device=device)
        # 位置编码不需要梯度更新
        self.encoding.requires_grad = False
        # 创建一个从0到max_len-1的序列，用于表示位置
        pos = torch.arange(0, max_len)
        # 将位置序列转换为浮点数，并增加一个维度，形状变为(max_len, 1)
        pos = pos.float().unsqueeze(dim = 1)

        # 创建一个从0到d_model-1的序列，步长为2，用于计算正弦和余弦函数的参数
        _2i = torch.arange(0, d_model, step = 2).float()
        # 计算位置编码的偶数位置的值，使用正弦函数
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        # 计算位置编码的奇数位置的值，使用余弦函数
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :];

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, drop_prob: float, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p = drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return  self.drop_out(tok_emb + pos_emb)
