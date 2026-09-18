import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)              # [B, 64*7*7]
        x = F.relu(self.fc1(x))                # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                        # [B, 10]
        return x


if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")