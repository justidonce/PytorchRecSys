import torch
from torch import nn
from models import FactorizationMachine
from torch.optim import Adam


# 数据既有数值特征, 又有类别特征
xn = torch.randn(10, 4)  # 数值特征
xc1 = torch.randint(0, 5, size=(10,))  # 类别特征,类别总数 5
xc2 = torch.randint(0, 3, size=(10,))  # 类别特征,类别总数 3
xc = torch.stack((xc1, xc2), dim=1)
label = torch.randn(10, 1)

model = FactorizationMachine(4, [5, 3], k=8)
optimizer = Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
for i in range(100):
    y = model(xn, xc)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
