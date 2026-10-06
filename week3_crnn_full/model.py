import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=128, lstm_layers=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.MaxPool2d((2, 1)),
        )

        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_size,
                           num_layers=lstm_layers, bidirectional=True,
                           batch_first=False)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)
        return logits
