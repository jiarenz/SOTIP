import torch.nn as nn
import torch.nn.functional as F


class CohenModelSubspace(nn.Module):

    def __init__(self, ch_in: int = 10, ch_out: int = 2, **kwargs):
        super().__init__()
        hidden_size = kwargs.get('hidden_size', 512)

        self.fc1 = nn.Linear(ch_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, ch_out)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        x = self.fc4(x)
        return x