from gymnasium.core import RewardWrapper
import torch
import torch.nn as nn
import numpy as np


class RewardModelWrapper(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        model_path = '/home/hanne/aisf_project/sequence_model_results/first_attempt/reward_model.pth'
        self.model = ConvLSTMClassifier()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def reward(self, reward):
        if reward == 0:
            return 0
        model_input = np.stack(self.reduced_grids)
        model_input = zero_pad(model_input, pad=30)
        model_input = np.reshape(model_input, (1, *model_input.shape))
        model_input = torch.from_numpy(model_input.astype('float32'))
        model_input = torch.permute(model_input, (0, 1, 4, 2, 3))
        output = self.model(model_input)
        output = float(output.detach().cpu())
        decision = int(output > 0.5)
        return decision*reward


def zero_pad(sequence, pad=30):
    n_images, image_i, image_j, image_channels = sequence.shape
    zeros = np.zeros((pad, image_i, image_j, image_channels))
    zeros[-n_images:, :, :, :] = sequence
    return zeros


# Define Convolutional LSTM classifier model
class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        sequence_length = 30,
        input_size = 12,
        input_channels = 3,
        filters_mid = 16,
        filters_final = 4,
        kernel_size = 3,
        lstm_hidden_size = 64,
        lstm_n_layers = 2
    ):
        super(ConvLSTMClassifier, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.scaled_input_size = input_size//4
        self.sequence_length = sequence_length
        self.filters_final = filters_final
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_n_layers = lstm_n_layers

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            input_channels,
            filters_mid,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (6, 6)
        self.conv2 = nn.Conv2d(
            filters_mid,
            filters_final,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (3,3)
    
        lstm_input_size = filters_final*self.scaled_input_size**2
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_n_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = x.view(-1, self.input_channels, self.input_size, self.input_size)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(-1, self.sequence_length, self.filters_final, self.scaled_input_size, self.scaled_input_size)
        x = x.reshape(-1, self.sequence_length, self.filters_final*self.scaled_input_size*self.scaled_input_size)

        h0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.flatten(self.sigmoid(out))
        return out