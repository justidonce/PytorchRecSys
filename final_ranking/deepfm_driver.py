import torch
from torch import nn
from models import DeepFM
from torch.optim import Adam


xc1 = torch.randint(0, 5, size=(10, 1))
xc2 = torch.randint(0, 7, size=(10, 1))
xc3 = torch.randint(0, 8, size=(10, 1))
xc4 = torch.randint(0, 4, size=(10, 1))
xc = torch.concat((xc1, xc2, xc3, xc4), dim=1)
label = torch.randn(10, 1)

model = DeepFM([5, 7, 8, 4])
model.train()
optimizer = Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

for i in range(100):
    y = model(xc)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
