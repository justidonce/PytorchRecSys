import torch
from torch import nn
import torch.nn.functional as F


class DIN(nn.Module):
    """
    Deep Interest Network
    """

    def __init__(self, num_users, num_items, embed_dim=64, hidden_units=None, dropout_rate=0.1):
        """
        :param num_users: int, 用户总数
        :param num_items: int, 物品总数
        :param embed_dim: int, embedding 维度
        :param hidden_units: list of int, 隐藏层神经元个数
        :param dropout_rate: float
        """

        super().__init__()

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        if hidden_units is None:
            hidden_units = [64, 32]

        # attention layer
        self.attention = Attention(embed_dim, dropout_rate)

        # mlp
        self.fc = nn.Sequential()
        input_dim = 3 * embed_dim
        for unit in hidden_units:
            module = nn.Sequential(nn.Linear(input_dim, unit), Dice(unit), nn.Dropout(dropout_rate))
            self.fc.extend(module)
            input_dim = unit
        self.fc.append(nn.Linear(hidden_units[-1], 1))

    def forward(self, user_id, candidate_id, hist_seq, hist_mask):
        """
        :param user_id: 用户 id, (B, )
        :param candidate_id: 候选物品 id, (B, )
        :param hist_seq: 用户行为序列, (B, L)
        :param hist_mask: 用户行为序列 mask, (B, L)
        :return: logit, 输出特征向量, (B, 1)
        """

        user_embed = self.user_embed(user_id)  # (B, d)
        candidate_embed = self.item_embed(candidate_id)  # (B, d)
        hist_embed = self.item_embed(hist_seq)  # (B, L, d)

        attn_weights = self.attention(candidate_embed, hist_embed, hist_mask)  # (B, L)
        # 计算用户兴趣向量
        user_interest = attn_weights.unsqueeze(dim=-1) * hist_embed  # (B, L, 1) * (B, L, d) = (B, L, d)
        user_interest = torch.sum(user_interest, dim=1)  # (B, d)

        # 结合用户向量, 候选物品向量, 用户兴趣向量, 输出最终候选物品的特征向量
        embeddings = torch.concat((user_embed, candidate_embed, user_interest), dim=1)  # (B, 3d)
        logit = self.fc(embeddings)  # (B, 1)

        return logit


class Attention(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.1):
        super().__init__()

        # 使用 MLP 计算注意力
        self.mlp = nn.Sequential()
        input_dim = embed_dim * 4
        for i in range(2):
            hidden_dim = input_dim // 2
            module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                Dice(hidden_dim),
                nn.Dropout(dropout_rate)
            )
            self.mlp.extend(module)
            input_dim = hidden_dim
        self.mlp.append(nn.Linear(input_dim, 1))

    def forward(self, candidate_embed, hist_embed, hist_mask):
        """
        :param candidate_embed: 候选物品 embedding, (B, d)
        :param hist_embed: 用户行为序列 embedding, (B, L, d)
        :param hist_mask: 用户行为序列 mask, (B, L)
        :return: weights: 用户行为序列权重, (B, L)

        相当于 weights = Attention(query=candidate_embed, k=hist_embed, v=hist_embed)
        """

        candidate_embed = torch.unsqueeze(candidate_embed, dim=1)  # (B, 1, d)
        candidate_embed = torch.tile(candidate_embed, dims=(1, hist_embed.shape[1], 1))  # (B, L, d)

        # 融合候选物品向量, 历史序列向量, 两者的差值, 两者的乘积
        embeddings = torch.concat(
            (candidate_embed, hist_embed, candidate_embed - hist_embed, candidate_embed * hist_embed),
            dim=-1)  # (B, L, 4d)

        scores = self.mlp(embeddings)  # (B, L, 1)
        scores = torch.squeeze(scores, dim=-1)  # (B, L)

        # mask=0 的位置置为 -10^9, 让其权重为 0
        scores[hist_mask == 0] = -1e9
        weights = F.softmax(scores, dim=1)  # (B, L)

        return weights


class Dice(nn.Module):
    """
    DIN 使用的激活函数
    """

    def __init__(self, num_feature):
        super().__init__()

        self.alpha = nn.Parameter(torch.zeros(num_feature))  # (d, )
        self.bn = nn.BatchNorm1d(num_feature, affine=False)

    def forward(self, x):
        """
        :param x: 特征向量, (*, d)
        :return: y, 特征向量, (*, d)
        """

        assert x.dim() == 3 or x.dim() == 2
        # 序列输入需要转置一下维度
        if x.dim() == 3:
            x_norm = self.bn(x.transpose(1, 2))  # x: (B, L, d) -> (B, d, L), x_norm: (B, d, L)
            x_norm = torch.transpose(x_norm, 1, 2)  # (B, L, d)
        elif x.dim() == 2:
            x_norm = self.bn(x)  # (B, d)

        p = F.sigmoid(x_norm)  # (*, d)
        y = p * x + (1 - p) * self.alpha * x  # (*, d)

        return y
