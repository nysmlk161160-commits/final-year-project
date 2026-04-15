import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, c_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))

