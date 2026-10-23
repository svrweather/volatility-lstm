import torch.nn as nn
from src.my_model import MyLSTM


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, custom=False):
        super().__init__()
        self.lstm = (MyLSTM(input_dim, hidden_dim, num_layers) if custom 
                     else nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
