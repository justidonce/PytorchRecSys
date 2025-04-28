import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam


class Item2Vec(nn.Module):
    def __init__(self, item_vocab_size, embed_dim=128):
        super().__init__()

        self.item_embed = nn.Embedding(item_vocab_size, embed_dim)
        self.context_embed = nn.Embedding(item_vocab_size, embed_dim)

    def forward(self, centers, contexts, neg_contexts):
        """
        根据正负样本对计算 negative sampling loss
        正样本: (centers_i, contexts_i)
        负样本: (centers_i, neg_contexts_{ij}), j = 0, 1, ..., k-1
        :param centers: (B, )
        :param contexts: (B, )
        :param neg_contexts: (B, k)
        :return: loss: (1, )

        neg_loss = ln(1 + exp(-logit^{+})) + \sum_k ln(1 + exp(logit^{-}))
        logit^{+} = center_embed @ context_embed, 向量内积
        logit^{-} = center_embed @ neg_context_embed, 向量内积
        """

        center_embed = self.item_embed(centers)  # (B, d)
        context_embed = self.context_embed(contexts)  # (B, d)
        neg_context_embed = self.context_embed(neg_contexts)  # (B, k, d)

        pos_logit = torch.sum(center_embed * context_embed, dim=1)  # (B, )
        neg_logit = neg_context_embed @ center_embed.unsqueeze(dim=2)  # (B, k, d) @ (B, d, 1) = (B, k, 1)
        neg_logit = torch.squeeze(neg_logit, dim=2)  # (B, k)

        pos_loss = torch.log(1 + torch.exp(-pos_logit))  # (B, )
        neg_loss = torch.sum(torch.log(1 + torch.exp(neg_logit)), dim=1)  # (B, )
        loss = torch.mean(pos_loss + neg_loss)  # (1, )

        return loss


item_vocab_size = 10000  # 物品种数, V
max_seq_len = 50    # 最大序列长度
num_users = 10  # 用户总数, N
window = 5  # 滑动窗口大小
num_neg_sample = 8  # 负样本个数, k

# 生成用户行为序列, seq, seq_len
seq_len = torch.randint(2, max_seq_len, size=(num_users,))  # (N, )
seq = []
item_count = {item: 0 for item in range(item_vocab_size)}  # 统计物品出现的次数
for i in range(num_users):
    length = seq_len[i].item()
    session = torch.randint(item_vocab_size, size=(length,))

    for good in session:
        item_count[good.item()] += 1

    pad_length = max_seq_len - length
    padding = torch.zeros(pad_length, dtype=torch.int64)

    session = torch.concat((session, padding), dim=0)
    seq.append(session)

seq = torch.stack(seq, dim=0)  # (N, L)
item_prob = torch.tensor([count for count in item_count.values()])  # (V, )
item_prob = item_prob ** (3 / 4)  # p = n^{3/4}, 计算负样本的采样概率

# 生成正负样本
# 正样本: (center_i, context_i)
# 负样本: (center_i, neg_context_i), 每一个中心物品对应 k 个负样本物品
centers = []
contexts = []
neg_contexts = []

# center: x_i
# context: x[i-w: i-1] + x[i+1: i+w]
# neg_context: 在所有物品集合中, 根据频率随机采样, 抽取 k 个负样本物品
# 理想情况下, center: (2w, ), context: (2w, ), neg_context: (2w, k)
for session, session_len in zip(seq, seq_len):
    for i in range(session_len.item()):
        item = session[i]
        start = max(0, i - window)
        end = min(session_len.item(), i + window + 1)
        context = torch.concat((session[start: i], session[i + 1: end]), dim=0)
        center = torch.full(context.shape, fill_value=item)

        # 计算负样本的采样概率
        # 如果物品是上下文中的物品, 概率设为非常小的数, 让负样本中的物品大概率不是正样本中的物品
        sample_prob = item_prob.clone()
        sample_prob[context] = 1e-9
        sample_prob = sample_prob / torch.sum(sample_prob)
        neg_context = torch.multinomial(sample_prob, num_samples=num_neg_sample * context.numel(), replacement=True)
        neg_context = torch.reshape(neg_context, (-1, num_neg_sample))  # (context_size, k)

        centers.append(center)
        contexts.append(context)
        neg_contexts.append(neg_context)

# 所有用户的行为序列, 展开得到的正负样本
centers = torch.concat(centers, dim=0)  # (M, )
contexts = torch.concat(contexts, dim=0)  # (M, )
neg_contexts = torch.concat(neg_contexts, dim=0)  # (M, k)

dataset = TensorDataset(centers, contexts, neg_contexts)
dataloader = DataLoader(dataset, batch_size=16)

model = Item2Vec(item_vocab_size)
optimizer = Adam(model.parameters(), lr=0.01)

# 训练
for i in range(10):
    for batch in dataloader:
        batch_centers, batch_contexts, batch_neg_contexts = batch
        loss = model(batch_centers, batch_contexts, batch_neg_contexts)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 物品的 embedding 等于 item, context embedding 的平均值
with torch.no_grad():
    trained_embeddings = (model.item_embed.weight + model.context_embed.weight) / 2
    print(trained_embeddings.shape)
