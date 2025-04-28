import torch
from torch import nn


class WideDeep(nn.Module):
    def __init__(
            self,
            wide_num_feature,
            deep_num_feature,
            num_classes,
            hidden_units=None,
            embed_dim=16,
            dropout_rate=0.1
    ):
        """
        :param wide_num_feature: int, wide 端的特征个数, d_w
        :param deep_num_feature: int, deep 端的数值特征个数, n1
        :param num_classes: list of int, deep 端的类别特征的类别总数, 长度为 n2
        :param hidden_units: list of int, DNN 隐藏层单元数
        :param embed_dim: int, deep 端的类别特征的嵌入维度
        :param dropout_rate: float, dropout 概率
        """

        super().__init__()

        # wide 网络是简单的一阶项
        self.wide = nn.Linear(wide_num_feature, 1)

        self.deep_numerical = nn.BatchNorm1d(deep_num_feature)
        self.deep_categorical = nn.ModuleList()
        for num_class in num_classes:
            self.deep_categorical.append(nn.Embedding(num_class, embed_dim))

        if hidden_units is None:
            hidden_units = [256, 128]

        # deep 网络是一个多层 MLP
        self.deep = nn.Sequential()
        input_dim = embed_dim * len(num_classes) + deep_num_feature
        for unit in hidden_units:
            module = nn.Sequential(
                nn.Linear(input_dim, unit),
                nn.BatchNorm1d(unit),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.deep.extend(module)
            input_dim = unit
        self.deep.append(nn.Linear(input_dim, 1))

    def forward(self, xw, xn, xc):
        """
        :param xw: wide 端的特征向量, (B, d_w)
        :param xn: deep 端的数值特征向量, (B, n1)
        :param xc: deep 端的类别特征向量, (B, n2)
        :return: logit, 输出特征向量, (B, 1)
        """

        wide_logit = self.wide(xw)  # (B, 1)

        numerical_embed = self.deep_numerical(xn)  # (B, n1)
        categorical_embed = []
        for i in range(xc.shape[1]):
            module = self.deep_categorical[i]
            categorical_embed.append(module(xc[:, i]))  # (B, embed_dim)
        categorical_embed = torch.concat(categorical_embed, dim=1)  # (B, n2 * embed_dim)

        deep_embed = torch.concat((numerical_embed, categorical_embed), dim=1)  # (B, n1 + n2 * embed_dim)
        deep_logit = self.deep(deep_embed)  # (B, 1)
        logit = wide_logit + deep_logit  # (B, 1)

        return logit
