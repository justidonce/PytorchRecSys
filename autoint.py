import torch
from torch import nn
import torch.nn.functional as F


class AutoInt(nn.Module):
    def __init__(self, num_feature, num_classes, num_layers=3, num_heads=2, embed_dim=64):
        super().__init__()

        self.numerical_embed = nn.Parameter(torch.randn(num_feature, embed_dim))

        self.categorical_embed = nn.ModuleList()
        for num_class in num_classes:
            self.categorical_embed.append(nn.Embedding(num_class, embed_dim))

        self.mha = nn.ModuleList([MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)])

        seq_len = num_feature + len(num_classes)
        self.output_layer = nn.Linear(seq_len * embed_dim, 1)

    def forward(self, xn, xc):
        """
        :param xn: 数值特征, (B, n1)
        :param xc: 类别特征, (B, n2)
        :return: logit, 输出特征, (B, 1)
        """

        batch_size = xn.shape[0]

        xn_embed = xn[:, :, None] * self.numerical_embed[None]  # (B, n1, 1) .* (1, n1, d) = (B, n1, d)

        xc_embed = []
        for i in range(xc.shape[1]):
            module = self.categorical_embed[i]
            xc_embed.append(module(xc[:, i]))  # (B, d)
        xc_embed = torch.stack(xc_embed, dim=1)  # (B, n2, d)

        embeddings = torch.concat((xn_embed, xc_embed), dim=1)  # (B, n1+n2, d)
        for module in self.mha:
            embeddings = module(embeddings)  # (B, n1+n2, d)

        embeddings = torch.reshape(embeddings, (batch_size, -1))  # (B, (n1+n2) * d)
        logit = self.output_layer(embeddings)  # (B, 1)

        return logit


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: int, embedding 维度
        :param num_heads: int, head 个数
        """

        super().__init__()
        self.multihead = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        :param x: 特征向量, (B, n, d)
        :return: self attention 后的特征向量, (B, n, d)
        """

        x_attn, _ = self.multihead(query=x, key=x, value=x, need_weights=False)  # (B, n, d)
        x = x + x_attn  # (B, n, d)
        x = F.relu(x)  # (B, n, d)

        return x
