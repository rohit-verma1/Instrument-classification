from torch import nn
from torchsummary import summary
import torch


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(10368, 14)

    def forward(self, input_data):
        
        x = self.conv1(input_data)
        
        x = self.conv2(x)
        


        
        x = self.flatten(x)
        
        logits = self.linear(x)

        
        return logits


if __name__ == "__main__":

    cnn = CNN()
    summary(cnn, (1, 128, 44))
  
"""
    # Forward pass with a dummy input
    dummy_data = torch.randn(1, 128 , 44)  # Adjust based on your input size
    output_feature_map = cnn(dummy_data)

    # Print the shape of the feature map
    print(output_feature_map.shape)
"""    

