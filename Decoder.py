import torch
from Attention import *

class DecoderLayer(nn.Moduler):
    def __init__(self, d_model, num_head, ffn_hidden, dropout):
        """
        初始化 DecoderLayer 类。

        :param d_model: 输入和输出的特征维度。
        :param num_head: 多头注意力机制中的头数。
        :param ffn_hidden: 前馈网络中的隐藏层维度。
        :param dropout: dropout 概率。
        """
        super(DecoderLayer, self).__init__()
        # 初始化多头自注意力机制
        self.self_attention = MultiHeadAttention(d_model, num_head)
        # 初始化层归一化
        self.norm1 = LayerNorm(d_model)
        # 初始化 dropout 层
        self.dropout1 = nn.Dropout(p=dropout)

        # 初始化编码器-解码器注意力机制
        self.enc_dec_attention = MultiHeadAttention(d_model, num_head)
        # 初始化层归一化
        self.norm2 = LayerNorm(d_model)
        # 初始化 dropout 层
        self.dropout2 = nn.Dropout(p=dropout)

        # 初始化前馈网络
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        # 初始化层归一化
        self.norm3 = LayerNorm(d_model)
        # 初始化 dropout 层
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        前向传播过程。

        :param x: 解码器的输入张量，形状为 (batch_size, target_length, d_model)。
        :param enc_out: 编码器的输出张量，形状为 (batch_size, source_length, d_model)。
        :param trg_mask: 目标序列的掩码张量，形状为 (batch_size, target_length, target_length)。
        :param src_mask: 源序列的掩码张量，形状为 (batch_size, target_length, source_length)。
        :return: 经过解码器层处理后的输出张量，形状为 (batch_size, target_length, d_model)。
        """
        # 保存输入张量，用于残差连接
        _x = x
        # 进行多头自注意力机制计算
        x = self.self_attention(q=x, k=x, v=x, mask=trg_mask)
        # 对注意力机制的输出进行dropout操作
        x = self.dropout1(x)
        # 进行残差连接和层归一化
        x = self.norm1(x + _x)

        # 如果编码器的输出不为空，则进行编码器-解码器注意力机制计算
        if enc_out is not None:
            # 保存输入张量，用于残差连接
            _x = x
            # 进行编码器-解码器注意力机制计算
            x = self.enc_dec_attention(q=x, k=enc_out, v=enc_out, mask=src_mask)
            # 对注意力机制的输出进行dropout操作
            x = self.dropout2(x)
            # 进行残差连接和层归一化
            x = self.norm2(x + _x)
        # 保存输入张量，用于残差连接
        _x = x
        # 进行前馈网络计算
        x = self.ffn(x)
        # 对前馈网络的输出进行dropout操作
        x = self.dropout3(x)
        # 进行残差连接和层归一化
        x = self.norm3(x + _x)
        # 返回处理后的输出张量
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, num_head, num_layer, drop_prob, device):
        """
        初始化 Decoder 类。

        :param dec_voc_size: 解码器词汇表的大小。
        :param max_len: 输入序列的最大长度。
        :param d_model: 模型的隐藏层维度。
        :param ffn_hidden: 前馈网络中的隐藏层维度。
        :param num_head: 多头注意力机制中的头数。
        :param num_layer: 解码器层的数量。
        :param drop_prob: dropout 概率。
        :param device: 模型运行的设备。
        """
        super(Decoder, self).__init__()
        # 初始化 Transformer 嵌入层
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)
        # 创建多个 DecoderLayer 并添加到 ModuleList 中
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  num_head=num_head,
                                                  ffn_hidden=ffn_hidden,
                                                  dropout=drop_prob)
                                     for _ in range(num_layer)])
        
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        前向传播过程。

        :param trg: 解码器的输入张量，形状为 (batch_size, target_length)。
        :param enc_out: 编码器的输出张量，形状为 (batch_size, source_length, d_model)。
        :param trg_mask: 目标序列的掩码张量，形状为 (batch_size, target_length, target_length)。
        :param src_mask: 源序列的掩码张量，形状为 (batch_size, target_length, source_length)。
        :return: 经过解码器处理后的输出张量，形状为 (batch_size, target_length, dec_voc_size)。
        """
        # 对解码器的输入进行嵌入
        x = self.emb(x)
        # 对每个解码器层进行前向传播
        for layer in self.layers:
            x = layer(x, enc_out, trg_mask, src_mask)
        # 对输出进行线性变换
        x = self.linear(x)

        return x