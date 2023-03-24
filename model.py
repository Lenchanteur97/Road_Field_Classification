import torch
import torch.nn as nn
import torch.nn.functional as f

import os


class CustomBinaryClassifier(nn.Module):

    def __init__(self, saving_suffix=''):
        super(CustomBinaryClassifier, self).__init__()
        self.saving_suffix = saving_suffix  # Suffix for the filename of model weights
        # input size : 64*64
        self.conv1 = self.conv_block(in_ch=3, out_ch=8, kernel_size=3, padding=1)  # 32*32
        self.conv2 = self.conv_block(in_ch=8, out_ch=16, kernel_size=3, padding=1)  # 16*16
        self.conv3 = self.conv_block(in_ch=16, out_ch=32, kernel_size=3, padding=1)  # 8*8
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def conv_block(self, in_ch, out_ch, kernel_size, padding):
        """
        Create a convolution block for the model

        :param in_ch: Number of channels of the input
        :param out_ch: Number of channels of the output
        :param kernel_size: Kernel size
        :param padding: Amount of padding applied to the input
        :return: A convolution block composed of a 2D convolution layer followed by ReLU activation and max pooling
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x

    def save_model(self):
        """
        Save the layers parameters into working directory
        """
        torch.save(self.state_dict(), os.path.join(os.getcwd(), 'CustomCNN_best'+str(self.saving_suffix)+'.pth.tar'))

    def load_best_model(self):
        """
        Load the best layers parameters learned during training
        """
        state_dict = torch.load(os.path.join(os.getcwd(), 'CustomCNN_best'+str(self.saving_suffix)+'.pth.tar'), map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
