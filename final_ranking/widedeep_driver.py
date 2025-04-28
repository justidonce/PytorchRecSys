import torch
from torch import nn
from models import WideDeep
from torch.optim import Adam


xw = torch.randn(10, 5)
xn = torch.randn(10, 6)
xc1 = torch.randint(0, 5, size=(10, 1))
xc2 = torch.randint(0, 3, size=(10, 1))
xc = torch.concat((xc1, xc2), dim=1)
label = torch.rand(10, 1)

model = WideDeep(5, 6, [5, 3])
model.train()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(100):
    y = model(xw, xn, xc)
    loss = criterion(y, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
