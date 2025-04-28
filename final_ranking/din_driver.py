import torch
from torch import nn
from models import DIN
from torch.optim import Adam


# 数据生成, 用户 id, 候选物品 id, 历史物品序列
batch_size = 32
num_users = 100
num_items = 50
max_len = 20

user_id = torch.randint(0, num_users, size=(batch_size,))
candidate_id = torch.randint(0, num_items, size=(batch_size,))

hist_seq = []
hist_mask = []
hist_len = torch.randint(1, max_len + 1, size=(batch_size,))  # 历史序列中至少有一个物品
for i in range(batch_size):
    seq_len = hist_len[i].item()
    seq = torch.randint(0, num_items, size=(1, seq_len))
    pad_len = max_len - seq_len
    pad = torch.zeros(1, pad_len)
    seq = torch.concat((seq, pad), dim=1)
    mask = torch.ones(1, seq_len)
    mask = torch.concat((mask, pad), dim=1)

    hist_seq.append(seq)
    hist_mask.append(mask)

hist_seq = torch.concat(hist_seq, dim=0).long()
hist_mask = torch.concat(hist_mask, dim=0)
label = torch.randint(0, 2, size=(batch_size, 1)).float()

model = DIN(num_users, num_items)
model.train()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for i in range(100):
    y = model(user_id, candidate_id, hist_seq, hist_mask)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
