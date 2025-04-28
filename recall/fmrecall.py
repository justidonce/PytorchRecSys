import torch
from torch import nn
from torch.optim import Adam


class FMRecall(nn.Module):
    def __init__(self, num_users, item_vocab_size, user_age_bin, item_cat, embed_dim=128):
        super().__init__()

        self.user_id_weight = nn.Embedding(num_users, 1)
        self.user_age_weight = nn.Embedding(user_age_bin, 1)
        self.item_id_weight = nn.Embedding(item_vocab_size, 1)
        self.item_cat_weight = nn.Embedding(item_cat, 1)

        self.user_id_embed = nn.Embedding(num_users, embed_dim)
        self.user_age_embed = nn.Embedding(user_age_bin, embed_dim)
        self.item_id_embed = nn.Embedding(item_vocab_size, embed_dim)
        self.item_cat_embed = nn.Embedding(item_cat, embed_dim)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, user_age, item_id, item_cat):
        """
        根据用户特征, 物品特征计算 score
        score 越大, 该物品越有可能被用户点击

        :param user_id: 用户 id, (B, )
        :param user_age: 用户年龄区间, (B, )
        :param item_id: 物品 id, (B, )
        :param item_cat: 物品类别, (B, )
        :return: score, (B, )

        score = b + w_u + w_t + <v_u, v_t>, 实际上就是用户特征和物品特征的 FM logit
        v_u = u_1 + u_2 + ..., 所有用户特征的和, 在测试时直接当做用户特征的 embedding
        v_t = v_1 + v_2 + ..., 所有物品特征的和, 在测试时直接当做物品特征的 embedding
        召回时, 根据 v_u, v_t 两个向量计算相似度, v_t 可以离线存储
        """

        user_id_weight = self.user_id_weight(user_id)  # (B, 1), w_u
        user_age_weight = self.user_age_weight(user_age)  # (B, 1), w_u
        item_id_weight = self.item_id_weight(item_id)  # (B, 1), w_t
        item_cat_weight = self.item_cat_weight(item_cat)  # (B, 1), w_t
        linear_part = user_id_weight + user_age_weight + item_id_weight + item_cat_weight  # (B, 1), w_u + w_t

        user_id_embed = self.user_id_embed(user_id)  # (B, d)
        user_age_embed = self.user_age_embed(user_age)  # (B, d)
        item_id_embed = self.item_id_embed(item_id)  # (B, d)
        item_cat_embed = self.item_cat_embed(item_cat)  # (B, d)
        user_embed = user_id_embed + user_age_embed  # (B, d), v_u
        item_embed = item_id_embed + item_cat_embed  # (B, d), v_t
        intersection_part = torch.sum(user_embed * item_embed, dim=1, keepdim=True)  # (B, 1), v_u * v_t

        score = linear_part + intersection_part + self.bias  # (B, 1), b + w_u + w_t + v_u * v_t
        score = torch.squeeze(score, dim=1)  # (B, )

        return score


num_users = 1000  # 用户总数
item_vocab_size = 5000  # 物品种数
user_age_bin = 5  # 用户年龄区间个数
item_cat = 30  # 物品类别个数
batch_size = 10

# User to Item 的 FM 召回
# 正样本: (u, t^{+}), 用户和点击过的物料
# 负样本: (u, t^{-}), 随机采样, 根据物料出现的频率
# 正样本, 负样本都为 B 个
user_id = torch.randint(num_users, size=(batch_size,))
user_age = torch.randint(user_age_bin, size=(batch_size,))
click_item_id = torch.randint(item_vocab_size, size=(batch_size,))
click_item_cat = torch.randint(item_cat, size=(batch_size,))

# 使用指数分布模拟物品出现的频率
rate = 1.0
exp_dist = torch.distributions.Exponential(rate)
item_freq = exp_dist.sample(torch.Size([item_vocab_size, ]))
item_freq = rate * torch.exp(-rate * item_freq)  # p = lambda * e^{-lambda * x}
item_freq = item_freq / torch.sum(item_freq)
neg_item_id = torch.multinomial(item_freq, num_samples=batch_size, replacement=True)
neg_item_cat = torch.randint(item_cat, size=(batch_size,))

model = FMRecall(num_users, item_vocab_size, user_age_bin, item_cat)
optimizer = Adam(model.parameters(), lr=0.001)

for i in range(100):
    pos_score = model(user_id, user_age, click_item_id, click_item_cat)
    neg_score = model(user_id, user_age, neg_item_id, neg_item_cat)
    loss = torch.log(1 + torch.exp(neg_score - pos_score))  # BPR Loss
    loss = torch.mean(loss)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
