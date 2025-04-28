import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from models import AutoInt


xn = torch.randn(10, 4)
xc1 = torch.randint(0, 5, size=(10, 1))
xc2 = torch.randint(0, 3, size=(10, 1))
xc = torch.concat((xc1, xc2), dim=1)
label = torch.randn(10, 1)

model = AutoInt(4, [5, 3])
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(100):
    y = model(xn, xc)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
