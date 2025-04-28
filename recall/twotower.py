import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from collections import defaultdict
import random


class Shopping(Dataset):
    def __init__(self, num_users, item_vocab_size, max_seq_len, dim_user_feature=64, dim_item_feature=64):
        self.num_users = num_users
        self.user_pos_item = defaultdict(list)  # 用户: [用户点击过的物品]
        self.data_pairs = []  # user_pos_item 的展开, [(用户, 点击过的物品)]
        self.item_count = [0] * item_vocab_size  # 物品出现的次数
        self.user_features = torch.randn(num_users, dim_user_feature)  # 除了用户 id 外的其余用户特征
        self.item_features = torch.randn(item_vocab_size, dim_item_feature)  # 除了物品 id 外的其余物品特征

        for user_id in range(num_users):
            seq_len = random.randint(1, max_seq_len)  # 用户点击物品的个数
            seq = random.sample(range(item_vocab_size), seq_len)
            self.user_pos_item[user_id] = seq

            pairs = []
            for item_id in seq:
                pairs.append((user_id, item_id))
                self.item_count[item_id] += 1
            self.data_pairs.extend(pairs)

    def __getitem__(self, index):
        user_id, item_id = self.data_pairs[index]
        user_id = torch.tensor(user_id, dtype=torch.int64)
        item_id = torch.tensor(item_id, dtype=torch.int64)
        user_feature = self.user_features[user_id]
        item_feature = self.item_features[item_id]

        return user_id, item_id, user_feature, item_feature


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__()

        self.batch_size = batch_size
        self.user_ids = list(range(dataset.num_users))  # [user_id]
        self.user_pair_indices = defaultdict(list)  # user_id: [data_pair_index]

        if shuffle:
            random.shuffle(self.user_ids)

        for i, pair in enumerate(dataset.data_pairs):
            user_id, _ = pair
            self.user_pair_indices[user_id].append(i)

    def __iter__(self):
        """
        每次 for batch in dataloader, 返回一个用户均匀分布的 batch

        :return: batch, [data_pair_ind1, data_pair_ind2, ...], (B, )
        每个 data_pair_ind 都对应于不同的用户, 所以说是用户均匀分布的 batch sampler
        """

        batch = []
        for user_id in self.user_ids:
            pair_indices = self.user_pair_indices[user_id]  # 用户对应的 data_pair_ind 列表
            ind = random.choice(pair_indices)  # 随机选择一个 ind 作为用户点击过的物品, 即正样本
            batch.append(ind)

            # batch 个数满足后, 返回 batch
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()  # 清空列表, 继续填装样本

        # 如果最后一个 batch 的长度小于 batch_size, 且不为空
        # 那么仍然返回 batch
        if batch:
            yield batch


class TwoTower(nn.Module):
    def __init__(self, user_tower, item_tower):
        """
        双塔召回网络

        :param user_tower: 用户塔
        :param item_tower: 物品塔
        """
        super().__init__()

        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_id, item_id, user_feature, item_feature):
        """
        汇集用户塔, 物品塔的计算结果

        :param user_id: 用户 id, (B, )
        :param item_id: 物品 id, (B, )
        :param user_feature: 用户其余特征, (B, d1)
        :param item_feature: 物品其余特征, (B, d2)
        :return:
        user_embed: 用户 embedding, (B, d)
        item_embed: 物品 embedding, (B, d)
        """

        user_embed = self.user_tower(user_id, user_feature)  # (B, d)
        item_embed = self.item_tower(item_id, item_feature)  # (B, d)

        return user_embed, item_embed


class UserTower(nn.Module):
    def __init__(self, num_users, dim_feature, embed_dim=64):
        super().__init__()

        self.user_embed = nn.Embedding(num_users, embed_dim)  # 处理用户 id
        self.mlp = nn.Sequential(
            nn.Linear(dim_feature, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, embed_dim),
            nn.Dropout(0.1),
        )  # 使用 dnn 处理用户其余特征

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)


    def forward(self, user_id, user_feature):
        """
        计算用户 embedding

        :param user_id: 用户 id, (B, )
        :param user_feature: 用户其余特征, (B, d1)
        :return: user_embed: 用户 embedding, (B, d)
        """

        user_embed = self.user_embed(user_id)  # (B, d)
        user_feature = self.mlp(user_feature)  # (B, d)
        user_embed = user_embed + user_feature  # (B, d)
        user_embed = F.normalize(user_embed, dim=1)  # (B, d)

        return user_embed


class ItemTower(nn.Module):
    def __init__(self, item_vocab_size, dim_feature, embed_dim=64):
        super().__init__()

        self.item_embed = nn.Embedding(item_vocab_size, embed_dim)  # 处理物品 id
        self.mlp = nn.Sequential(
            nn.Linear(dim_feature, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, embed_dim),
            nn.Dropout(0.1),
        )  # 使用 dnn 处理物品其余特征

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, item_id, item_feature):
        """
        计算物品 embedding

        :param item_id: 物品 id, (B, )
        :param item_feature: 物品其余特征, (B, d2)
        :return: item_embed: 物品 embedding, (B, d)
        """

        item_embed = self.item_embed(item_id)  # (B, d)
        item_feature = self.mlp(item_feature)  # (B, d)
        item_embed = item_embed + item_feature  # (B, d)
        item_embed = F.normalize(item_embed, dim=1)  # (B, d)

        return item_embed


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, dataset, temperature=0.5):
        """
        :param dataset: 数据集
        :param temperature: 温度系数, [0.0, 1.0]
        """

        super().__init__()

        self.temperature = temperature
        self.user_pos_item = {}  # user_id: [item_id]
        for user_id in dataset.user_pos_item.keys():
            item_ids = dataset.user_pos_item[user_id]
            self.user_pos_item[user_id] = torch.tensor(item_ids)  # 将点击过的物品 id 列表转化为张量

        self.item_freq = torch.tensor(dataset.item_count)  # 物品出现频率
        self.item_freq = self.item_freq / sum(self.item_freq)

    def forward(self, user_embed, item_embed, user_id, item_id):
        """
        使用 batch 负采样的方法, 计算 sampled softmax loss
        正样本: (用户, 用户点击过的某个物品)
        负样本: (用户, 其余用户点击过的物品)
        正样本有 1 个, 负样本有 <= B-1 个

        :param user_embed: 用户 embedding, (B, d)
        :param item_embed: 物品 embedding, (B, d)
        :param user_id: 用户 id, (B, )
        :param item_id: 物品 id, (B, )
        :return: loss, (1, )

        loss = -ln(Softmax(logit^{+}))
        logit(u, t) = (u * t) / T - log Q(t)
        Q(t) = freq(t)
        """

        batch_size = user_id.numel()
        mask = []

        # 创建 mask
        # 如果用户有多个点击的物品, 只有被 batch sampler 选择的物品 (对角线的物品) 设为正样本
        # 其余被点击的物品设为负样本, 让其 logit = -inf
        for user in user_id:
            pos_items = self.user_pos_item[user.item()]  # 用户点击过的物品
            pos_mask = torch.isin(item_id, pos_items)  # 点击过的物品设为负样本, mask = True, (B, )
            mask.append(pos_mask)
        mask = torch.stack(mask, dim=0)  # (B, B)
        mask[torch.arange(batch_size), torch.arange(batch_size)] = False  # 对角线 mask = False, 保证对角线为正样本

        logit = user_embed @ item_embed.t()  # (B, d) @ (d, B) = (B, B)
        logit = logit / self.temperature
        log_q = torch.log(self.item_freq[item_id])
        logit = logit - log_q  # (B, B), logit(u, t) = (u @ t) / T - log Q(t)

        logit[mask] = -1e9  # mask = False 的位置表示负样本, 防止其他点击物品对对角线上的正样本干扰
        # 直接用公式, log 数值不稳定, 容易出现 nan
        # loss = -torch.log(F.softmax(logit, dim=1))  # (B, B)
        # loss = torch.mean(torch.diag(loss))  # (1, )

        # 得使用 F.cross_entropy
        labels = torch.arange(batch_size, dtype=torch.int64)
        loss = F.cross_entropy(logit, labels)  # (1, )

        return loss


if __name__ == "__main__":
    num_users = 1000  # 用户总数
    item_vocab_size = 5000  # 物品种数
    max_seq_len = 30
    dim_user_feature = 64  # 用户其他特征的维度
    dim_item_feature = 64  # 物品其他特征的维度
    batch_size = 10

    dataset = Shopping(num_users, item_vocab_size, max_seq_len)
    batch_sampler = BalancedBatchSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    user_tower = UserTower(num_users, dim_user_feature)
    item_tower = ItemTower(item_vocab_size, dim_item_feature)
    model = TwoTower(user_tower, item_tower)
    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = SampledSoftmaxLoss(dataset)

    for i in range(10):
        for batch in dataloader:
            user_id, item_id, user_feature, item_feature = batch
            user_embed, item_embed = model(user_id, item_id, user_feature, item_feature)
            loss = criterion(user_embed, item_embed, user_id, item_id)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
