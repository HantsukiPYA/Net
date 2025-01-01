import torch
import torch.nn as nn
import math
from torch import Tensor
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        """
        初始化 ScaledDotProductAttention 类。

        这个初始化方法设置了注意力机制所需的 softmax 层。
        """
        # 调用父类 nn.Module 的初始化方法
        super(ScaledDotProductAttention, self).__init__()
        # 定义 softmax 层，用于计算注意力权重
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None, epsilon = 1e-12):
        batch_size, head, length, d_tensor = K.size()
        K_t = K.transpose(2, 3)
        score = (Q @ K_t) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)

        V = score @ V

        return V, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.attention = ScaledDotProductAttention()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.linear(d_model, d_model)
        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask = None):
        """
        多头注意力机制的前向传播过程。

        :param Q: 查询矩阵，形状为 (batch_size, length, d_model)。
        :param K: 键矩阵，形状为 (batch_size, length, d_model)。
        :param V: 值矩阵，形状为 (batch_size, length, d_model)。
        :param mask: 掩码矩阵，形状为 (batch_size, 1, length, length)，用于屏蔽不需要关注的位置。
        :return: 多头注意力机制的输出，形状为 (batch_size, length, d_model)。
        """
        # 对输入的查询、键和值矩阵进行线性变换
        Q = self.W_Q(Q), K = self.W_K(K), V = self.W_V(V)
        
        # 将查询、键和值矩阵分割成多个头
        Q = self.split(Q), K = self.split(K), V = self.split(V)

        # 计算缩放点积注意力
        out, attention = self.attention(Q, K, V, mask = mask)

        # 将多头注意力的输出拼接起来
        out = self.concat(out)
        # 对拼接后的输出进行线性变换
        out = self.W_concat(out)

        return out
    
    def split(self, tensor: Tensor):
        """
        将输入张量分割成多个头。

        :param tensor: 输入张量，形状为 (batch_size, length, d_model)。
        :return: 分割后的张量，形状为 (batch_size, head, d_tensor, length)。
        """
        # 获取输入张量的形状
        batch_size, length, d_model = tensor.size()
        # 计算每个头的维度
        d_tensor = d_model // self.num_head
        # 将输入张量重新调整形状，以便分割成多个头
        tensor = tensor.view(batch_size, self.num_head, d_tensor, length)
        # 转置张量，使头的维度成为第二维
        tensor = tensor.transpose(1, 2)
        return tensor
    
    def concat(self, tensor: Tensor):
        """
        将多头注意力的输出拼接起来。

        :param tensor: 输入张量，形状为 (batch_size, head, d_tensor, length)。
        :return: 拼接后的张量，形状为 (batch_size, length, d_model)。
        """
        # 获取输入张量的形状
        batch_size, head, length, d_tensor = tensor.size()
        # 计算拼接后的维度
        d_model = head * d_tensor
        # 转置张量，使头的维度成为第三维
        tensor = tensor.transpose(1, 2).contiguous()
        # 将张量重新调整形状，以便拼接
        tensor = tensor.view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, hidend_dim: int, drop_prob = 0.1):
        """
        初始化 PositionwiseFeedForward 类。

        :param d_model: 输入特征的维度。
        :param hidden_dim: 隐藏层的维度。
        """
        # 调用父类 nn.Module 的初始化方法
        super(PositionwiseFeedForward, self).__init__()
        # 定义两层线性变换
        self.linear1 = nn.Linear(d_model, hidend_dim)
        self.linear2 = nn.Linear(hidend_dim, d_model)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义 dropout 层
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        """
        前向传播过程。

        :param x: 输入张量，形状为 (batch_size, length, d_model)。
        :return: 经过位置前馈网络的输出张量，形状为 (batch_size, length, d_model)。
        """
        # 对输入进行线性变换
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x;

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        """
        前向传播过程。

        :param x: 输入张量，形状为 (batch_size, length, d_model)。
        :return: 经过层归一化的输出张量，形状为 (batch_size, length, d_model)。
        """
        # 计算输入张量在最后一个维度上的均值，并保持维度不变
        mean = x.mean(-1, keepdim=True)
        # 计算输入张量在最后一个维度上的方差，并保持维度不变，这里使用无偏估计
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' 表示最后一个维度。

        # 对输入张量进行归一化，减去均值并除以标准差
        out = (x - mean) / torch.sqrt(var + self.eps)
        # 对归一化后的张量进行缩放和偏移
        out = self.gamma * out + self.beta
        return out
