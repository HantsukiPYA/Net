import torch 
import torch.nn as nn

from Encoder import *
from Decoder import *

class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, max_len, d_model, ffn_hidden, num_head, num_layer, drop_prob, device):
        """
        初始化 Transformer 模型。

        参数:
        - enc_voc_size: 编码器词汇表的大小。
        - dec_voc_size: 解码器词汇表的大小。
        - max_len: 输入序列的最大长度。
        - d_model: 模型的隐藏层维度。
        - ffn_hidden: 前馈神经网络的隐藏层维度。
        - num_head: 多头注意力机制的头数。
        - num_layer: 编码器和解码器层的数量。
        - drop_prob: Dropout 概率。
        - device: 模型运行的设备。
        """
        super(Transformer, self).__init__()
        # 初始化编码器
        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               num_head=num_head,
                               num_layer=num_layer,
                               drop_prob=drop_prob,
                               device=device)
        # 初始化解码器
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               num_head=num_head,
                               num_layer=num_layer,
                               drop_prob=drop_prob,
                               device=device)
    def forward(self, src, trg):
        """
        前向传播过程。

        :param src: 源序列的输入张量，形状为 (batch_size, source_length)。
        :param trg: 目标序列的输入张量，形状为 (batch_size, target_length)。
        :return: 模型的输出张量，形状为 (batch_size, target_length, dec_voc_size)。
        """
        # 生成源序列的掩码张量
        src_mask = self.make_src_mask(src)
        # 生成目标序列的掩码张量
        trg_mask = self.make_trg_mask(trg)
        # 将源序列输入编码器，得到编码器的输出
        enc_src = self.encoder(src, src_mask)
        # 将目标序列、编码器的输出以及掩码张量输入解码器，得到解码器的输出
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        # 返回解码器的输出
        return output
    def make_src_mask(self, src):
        """
        生成源序列的掩码张量。

        :param src: 源序列的输入张量，形状为 (batch_size, source_length)。
        :return: 源序列的掩码张量，形状为 (batch_size, 1, 1, source_length)。
        """
        # 创建一个布尔张量，其中 True 表示源序列中不等于填充索引的位置，False 表示等于填充索引的位置
        # 使用 unsqueeze 方法在维度1和2上增加维度，以便与注意力机制中的张量形状匹配
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # 返回源序列的掩码张量
        return src_mask
    def make_trg_mask(self, trg):
        """
        生成目标序列的掩码张量。

        :param trg: 目标序列的输入张量，形状为 (batch_size, target_length)。
        :return: 目标序列的掩码张量，形状为 (batch_size, 1, target_length, target_length)。
        """
        # 创建一个布尔张量，其中 True 表示目标序列中不等于填充索引的位置，False 表示等于填充索引的位置
        # 使用 unsqueeze 方法在维度1和3上增加维度，以便与注意力机制中的张量形状匹配
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        # 获取目标序列的长度
        trg_len = trg.shape[1]
        # 创建一个下三角矩阵，其中主对角线及其以下的元素为1，其余元素为0
        # 将矩阵转换为ByteTensor类型，并将其移动到指定的设备上
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        
        # 将填充掩码和下三角掩码进行逐元素与操作，得到最终的目标序列掩码
        trg_mask = trg_pad_mask & trg_sub_mask
        # 返回目标序列的掩码张量
        return trg_mask