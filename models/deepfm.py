import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, num_classes, hidden_units=None, embed_dim=16, dropout_rate=0.1):
        """
        :param num_classes: list of int, 每个类别特征的类别总数, 长度为 n
        :param hidden_units: int, 隐藏层的神经元个数
        :param embed_dim: int
        :param dropout_rate: float
        """

        super().__init__()

        # FM 网络
        self.bias = nn.Parameter(torch.zeros(1))
        self.embed = nn.ModuleList()
        self.latent = nn.ModuleList()
        for num_class in num_classes:
            self.embed.append(nn.Embedding(num_class, 1))  # FM 一阶项权重
            self.latent.append(nn.Embedding(num_class, embed_dim))  # FM 的 embedding

        if hidden_units is None:
            hidden_units = [256, 128]

        # DNN 网络
        self.dnn = nn.Sequential()
        input_dim = embed_dim * len(num_classes)
        for unit in hidden_units:
            module = nn.Sequential(
                nn.Linear(input_dim, unit),
                nn.BatchNorm1d(unit),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.dnn.extend(module)
            input_dim = unit
        self.dnn.append(nn.Linear(input_dim, 1))

    def forward(self, xc):
        """
        :param xc: 类别特征向量, (B, n)
        :return: logit, 输出特征向量, (B, 1)
        """

        linear_part = torch.zeros(1).to(xc.device)  # FM 一阶项 sum(w_i)
        for i in range(xc.shape[1]):
            module = self.embed[i]
            linear_part = linear_part + module(xc[:, i])  # (B, 1)

        embeddings = []
        for i in range(xc.shape[1]):
            module = self.latent[i]
            embeddings.append(module(xc[:, i]))  # (B, d)

        # FM 二阶项
        embeddings_stack = torch.stack(embeddings, dim=1)  # (B, n, d)
        square_sum = embeddings_stack.sum(dim=1).pow(2).sum(dim=1)  # (B, )
        sum_square = embeddings_stack.pow(2).sum(dim=(1, 2))  # (B, )
        intersection_part = (square_sum - sum_square) / 2  # (B, )
        intersection_part = torch.unsqueeze(intersection_part, dim=1)  # (B, 1)

        # FM logit
        fm_logit = linear_part + intersection_part + self.bias

        # 将拼接的 FM embedding 输入到 DNN
        embeddings_concat = torch.concat(embeddings, dim=1)  # (B, nd)
        dnn_logit = self.dnn(embeddings_concat)  # (B, 1)

        logit = fm_logit + dnn_logit  # (B, 1)

        return logit
