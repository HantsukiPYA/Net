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
    