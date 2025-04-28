import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, num_feature, num_classes, k=10):
        """
        :param num_feature: int, 数值特征个数, n1
        :param num_classes: list of int, 每个类别的类别总数, 长度为 n2
        :param k: int, embedding v 的维度
        """

        super().__init__()

        # 数值特征网络
        self.numerical_fc = nn.Linear(num_feature, 1, bias=False)  # 一阶项, w_i * x_i
        self.numerical_latent = nn.ModuleList()  # linear 权重就是 embedding, [v_{i1}, v_{i2}, ...]
        for i in range(num_feature):
            self.numerical_latent.append(nn.Linear(1, k, bias=False))

        # 类别特征网络
        self.categorical_embed = nn.ModuleList()  # 一阶项, w_i * one hot = w_i
        self.categorical_latent = nn.ModuleList()  # embedding
        for num_class in num_classes:
            self.categorical_embed.append(nn.Embedding(num_class, 1))
            self.categorical_latent.append(nn.Embedding(num_class, k))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, xn, xc):
        """
        linear part = w_1 * x_{n1} + w_2 * x_{n2} + ... + w_{c1, 真实类别} + w_{c2, 真实类别} + ...
        intersection part = <v_{n1}, v_{n2}> * x_{n1} * x_{n2} + <v_{n1}, v_{n3}> * x_{n1} * x_{n3} + ...
                          + <v_{c1, 真实类别}, v_{c2, 真实类别}>  + <v_{c1, 真实类别}, v_{c3, 真实类别}>  + ...

        logit = linear part + intersection part + bias

        :param xn: 数值特征向量, (B, n1)
        :param xc: 类别特征向量, (B, n2)
        :return logit: 特征输出, (B, 1)
        """

        numerical_linear = self.numerical_fc(xn)  # (B, 1), w_1 * x_{n1} + w_2 * x_{n2} + ...
        categorical_linear = torch.zeros(1).to(xc.device)
        for i in range(xc.shape[1]):
            module = self.categorical_embed[i]
            categorical_linear = categorical_linear + module(xc[:, i])  # (B, 1), w_{c1, 真实类别} + w_{c2, 真实类别} + ...
        linear_part = numerical_linear + categorical_linear  # (B, 1)

        numerical_embedding = []
        for i in range(xn.shape[1]):
            module = self.numerical_latent[i]
            numerical_embedding.append(module(xn[:, i: i + 1]))  # (B, k), 实际上是 v_{ni} * x_{ni}
        numerical_embedding = torch.stack(numerical_embedding, dim=1)  # (B, n1, k)

        categorical_embedding = []
        for i in range(xc.shape[1]):
            module = self.categorical_latent[i]
            categorical_embedding.append(module(xc[:, i]))  # (B, k), v_{ci, 真实类别}
        categorical_embedding = torch.stack(categorical_embedding, dim=1)  # (B, n2, k)

        # 数值特征 embedding 拼接上类别特征 embedding, 二阶项计算可以统一操作 embedding
        embedding = torch.concat((numerical_embedding, categorical_embedding), dim=1)  # (B, n1 + n2, k)
        square_sum = embedding.sum(dim=1).pow(2).sum(dim=1)  # (B, ), embedding 分量的和的平方
        sum_square = embedding.pow(2).sum(dim=(1, 2))  # (B, ), embedding 各元素平方的和
        intersection_part = (square_sum - sum_square) / 2  # (B, ), 两者相减得到交叉项乘积
        intersection_part = torch.unsqueeze(intersection_part, dim=1)  # (B, 1)

        logit = linear_part + intersection_part + self.bias  # (B, 1)

        return logit
