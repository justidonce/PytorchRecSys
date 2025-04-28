import torch
from torch import nn


class DCNv1(nn.Module):
    def __init__(self, num_feature, num_layers, hidden_units=None, dropout_rate=0.1):
        """
        :param num_feature: int, 特征维度
        :param num_layers: int, cross network 网络层数, 层数为 n, 可以计算 n + 1 阶的交叉特征
        :param hidden_units: list of int, dnn 网络隐藏层神经元个数
        :param dropout_rate: float
        """

        super().__init__()

        # cross network
        self.cross_network = CrossNetworkV1(num_feature, num_layers)

        # dnn network
        if hidden_units is None:
            hidden_units = [64, 32]

        self.dnn = nn.Sequential()
        input_dim = num_feature
        for unit in hidden_units:
            module = nn.Sequential(nn.Linear(input_dim, unit), nn.ReLU(), nn.Dropout(dropout_rate))
            self.dnn.extend(module)
            input_dim = unit

        self.output_layer = nn.Linear(hidden_units[-1] + num_feature, 1)

    def forward(self, x):
        """
        :param x: 特征向量, (B, d)
        :return: logit, 输出特征向量, (B, d)
        """

        cross_out = self.cross_network(x)  # (B, d)
        dnn_out = self.dnn(x)  # (B, hidden_units[-1])

        # 拼接 cross network, dnn 两个网络输出的特征
        output = torch.concat((cross_out, dnn_out), dim=1)  # (B, d + hidden_units[-1])
        logit = self.output_layer(output)  # (B, 1)

        return logit


class DCNv2(nn.Module):
    def __init__(self, num_feature, num_layers, hidden_units=None, dropout_rate=0.1):
        """
        :param num_feature: int, 特征维度
        :param num_layers: int, cross network 层数, 如果有 n 层, 可以得到 n + 1 阶的交叉特征
        :param hidden_units: list of int, dnn 隐藏层神经元个数
        :param dropout_rate: float
        """

        super().__init__()

        # DCNv2 只是改变了 cross network, 增强了特征交叉的能力
        # 其余部分没有变化
        self.cross_network = CrossNetworkV2(num_feature, num_layers)

        if hidden_units is None:
            hidden_units = [64, 32]

        self.dnn = nn.Sequential()
        input_dim = num_feature
        for unit in hidden_units:
            module = nn.Sequential(nn.Linear(input_dim, unit), nn.ReLU(), nn.Dropout(dropout_rate))
            self.dnn.extend(module)
            input_dim = unit

        self.output_layer = nn.Linear(hidden_units[-1] + num_feature, 1)

    def forward(self, x):
        """
        :param x: 特征向量, (B, d)
        :return: logit, 输出特征向量, (B, 1)
        """

        cross_out = self.cross_network(x)  # (B, d)
        dnn_out = self.dnn(x)  # (B, hidden_units[-1])
        output = torch.concat((cross_out, dnn_out), dim=1)  # (B, hidden_units[-1] + d)
        logit = self.output_layer(output)  # (B, 1)

        return logit


class CrossNetworkV1(nn.Module):
    def __init__(self, num_feature, num_layers):
        """
        :param num_feature: int, 特征维度
        :param num_layers: int, 网络层数
        """

        super().__init__()

        self.num_feature = num_feature
        self.num_layers = num_layers
        self.weights = [nn.Parameter(torch.randn(num_feature, 1)) for _ in range(num_layers)]  # (d, 1)
        self.biases = [nn.Parameter(torch.randn(num_feature)) for _ in range(num_layers)]  # (d, )

    def forward(self, x):
        """
        迭代更新特征向量
        x_{i+1} = x_i + x_i @ w_i * x_0 + b_i

        :param x: 初始特征向量 x_0, (B, d)
        :return: 最终的特征向量 x_n, (B, d)
        """

        x0 = torch.clone(x)  # (B, d)
        for i in range(self.num_layers):
            xw = x @ self.weights[i]  # (B, 1)
            x = x + xw * x0 + self.biases[i]  # (B, d)

        return x


class CrossNetworkV2(nn.Module):
    def __init__(self, num_feature, num_layers):
        """
        :param num_feature: int, 特征维度
        :param num_layers: int, 网络层数
        """

        super().__init__()

        self.fcs = nn.ModuleList()
        for i in range(num_layers):
            self.fcs.append(nn.Linear(num_feature, num_feature))

    def forward(self, x):
        """
        迭代更新特征向量
        x_{i+1} = x_i + x_0 .* (W_i * x_i + b_i)

        :param x: 初始特征向量 x_0, (B, d)
        :return: 最终的特征向量 x_n, (B, d)
        """

        x0 = torch.clone(x)  # (B, d)
        for layer in self.fcs:
            xw = layer(x)  # (B, d)
            x = x + x0 * xw  # (B, d)

        return x
