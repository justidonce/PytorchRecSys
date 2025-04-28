import torch
from torch import nn
from models import DCNv2
from torch.optim import Adam


x = torch.randn(10, 12)
label = torch.randn(10, 1)

model = DCNv2(12, 3)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(100):
    y = model(x)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
