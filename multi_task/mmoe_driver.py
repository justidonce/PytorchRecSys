import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class Expert(nn.Module):
    def __init__(self, num_features, embed_dim):
        """
        专家网络负责特征提取, 融合
        """

        super().__init__()

        self.shallow = nn.Linear(num_features, embed_dim)
        self.deep = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        计算单个专家的输出特征
        :param x: 输入特征, (B, d_in)
        :return: output: 输出特征, (B, d_out)
        """

        output1 = self.shallow(x)  # (B, d_out)
        output2 = self.deep(x)  # (B, d_out)
        output = output1 + output2  # (B, d_out)

        return output


class Gate(nn.Module):
    def __init__(self, num_features, num_experts):
        """
        Gate 网络负责输出专家网络的权重
        """

        super().__init__()

        self.fc1 = nn.Linear(num_features, num_experts)
        self.fc2 = nn.Linear(num_experts, num_experts)

    def forward(self, x):
        """
        计算每个 expert 输出特征对应的权重
        :param x: 输入特征, (B, d_in)
        :return: weights: 专家权重, (B, n)
        """

        x = F.leaky_relu(self.fc1(x))  # (B, n)
        x = F.leaky_relu(self.fc2(x))  # (B, n)
        weights = F.softmax(x, dim=1)  # (B, n)

        return weights


class MultiGateMixtureOfExperts(nn.Module):
    def __init__(self, num_features, num_classes, num_experts=4, embed_dim=64):
        super().__init__()

        # 专家网络, 专家个数可以任意
        self.experts = nn.ModuleList([Expert(num_features, embed_dim) for _ in range(num_experts)])

        # 任务数为 2, 那么 gate 网络就有 2 个
        self.gates = nn.ModuleList([Gate(num_features, num_experts) for _ in range(2)])

        # 任务塔, 每个任务最终的映射头
        # 任务一是分类任务, 将维度映射到类别数
        # 任务二是回归任务, 将维度映射到 1
        self.towers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes),
            nn.Linear(embed_dim, 1)
        ])

    def forward(self, x):
        """
        :param x: 输入特征, (B, d_in)
        :return: outputs: 多任务的输出

        outputs = (logit, predict)
        logit 是分类任务的 logit,  (B, C)
        predict 是回归任务的预测值, (B, 1)
        """

        # 共享 embedding
        shared_embedding = [module(x) for module in self.experts]  # [(B, d_out)], n 个专家的输出特征
        shared_embedding = torch.stack(shared_embedding, dim=1)  # (B, n, d_out)

        weights_classify = self.gates[0](x)  # (B, n), 分类任务的权重
        weights_regress = self.gates[1](x)  # (B, n), 回归任务的权重

        # 专属于分类任务的 embedding
        # (B, n, 1) * (B, n, d_out) = (B, n, d_out)
        # u = w1 * e1 + w2 * e2 + ...
        classify_embedding = weights_classify.unsqueeze(dim=2) * shared_embedding
        classify_embedding = torch.sum(classify_embedding, dim=1)  # (B, d_out)

        # 专属于回归任务的 embedding
        regress_embedding = weights_regress.unsqueeze(dim=2) * shared_embedding  # (B, n, d_out)
        regress_embedding = torch.sum(regress_embedding, dim=1)  # (B, d_out)

        # 计算各自任务的输出
        classify_logit = self.towers[0](classify_embedding)  # (B, C)
        regress_predict = self.towers[1](regress_embedding)  # (B, 1)
        outputs = [classify_logit, regress_predict]

        return outputs


batch_size = 10
num_features = 20
num_classes = 5

input_feature = torch.randn(batch_size, num_features)
label1 = torch.randint(num_classes, size=(batch_size,))
label2 = torch.randn(batch_size, 1)

model = MultiGateMixtureOfExperts(num_features, num_classes)
optimizer = Adam(model.parameters(), lr=0.01)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

for i in range(10):
    outputs = model(input_feature)
    logit, predict = outputs
    loss1 = criterion1(logit, label1)
    loss2 = criterion2(predict, label2)
    loss_aggregate = 0.5 * loss1 + 0.5 * loss2
    print(loss_aggregate)

    optimizer.zero_grad()
    loss_aggregate.backward()
    optimizer.step()
