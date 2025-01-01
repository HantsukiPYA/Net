import torch
import torch.nn as nn
from Attention import *
from Embedding import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, ffn_hidden, dropout ):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = dropout) 

    def forward(self, x, mask):
        _x = x
        x = self.attention(q = x, k = x, v = x, mask = mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x;
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x;

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, num_head, num_layer, drop_prob, device):
        """
        初始化 Encoder 类。

        参数:
        - enc_voc_size: 编码器词汇表的大小。
        - max_len: 输入序列的最大长度。
        - d_model: 模型的隐藏层维度。
        - ffn_hidden: 前馈神经网络的隐藏层维度。
        - num_head: 多头注意力机制的头数。
        - num_layer: 编码器层的数量。
        - drop_prob: Dropout 概率。
        - device: 模型运行的设备。
        """
        super(Encoder, self).__init__()
        # 初始化 Transformer 嵌入层
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size,
                                        d_model=d_model, 
                                        max_len=max_len,
                                        drop_prob=drop_prob, 
                                        device=device)
        
        # 创建多个 EncoderLayer 并添加到 ModuleList 中
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                num_head = num_head,
                                                ffn_hidden = ffn_hidden,
                                                dropout = drop_prob)
                                                for _ in range(num_layer)])
    
    def forward(self, x, mask):
        x = self.emb(x)

        for layer in self.num_layer:
            x = layer(x, mask)
        
        return x