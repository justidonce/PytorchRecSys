import torch
from torch import nn
import torch.nn.functional as F
from optim import FTRL


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


x = torch.randn(8, 5)
y = torch.randint(0, 2, size=(8, 1), dtype=torch.float32)
model = Model()
optimizer = FTRL(model.parameters())
for _ in range(100):
    y_pred = model(x)
    loss = F.binary_cross_entropy(y_pred, y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
