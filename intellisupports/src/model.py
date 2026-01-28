# src/model.py

import torch
import torch.nn as nn

class TicketClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
