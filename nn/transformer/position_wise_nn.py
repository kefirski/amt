import torch.nn as nn

from .layer_norm import LayerNorm


class PositionWiseNN(nn.Module):
    def __init__(self, size, inner_size, dropout=0.1):
        super(PositionWiseNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size, inner_size),
            nn.SELU(),
            nn.Linear(inner_size, size)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, input):
        residual = input

        _, seq_len, size = input.size()
        input = input.view(-1, size)

        result = self.fc(input)
        result = result.view(-1, seq_len, size)
        result = self.dropout(result)

        return self.layer_norm(result + residual)
