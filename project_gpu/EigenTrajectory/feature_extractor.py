import torch
import torch.nn as nn
from torch.nn.functional import relu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class modified_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.e42 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.e52 = nn.Conv2d(128, 128, kernel_size=3, padding=0)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, frames):
        # Encoder
        outputs = []
        for frame in frames:
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            xe11 = relu(self.e11(frame))
            xe12 = relu(self.e12(xe11))
            xp1 = self.pool1(xe12)

            xe41 = relu(self.e41(xp1))
            xe42 = relu(self.e42(xe41))
            xp4 = self.pool4(xe42)

            xe51 = relu(self.e51(xp4))
            xe52 = relu(self.e52(xe51))

            # Global Average Pooling and Flatten
            pooled = self.global_avg_pool(xe52)
            flattened = pooled.view(pooled.size(0), -1)

            outputs.append(flattened)

        # Average the results
        avg_output = torch.stack(outputs).mean(0)

        return avg_output
