from collections import OrderedDict

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, device="cpu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        if device == "cuda" and torch.cuda.is_available():
            self.conv.cuda()
            self.batch_norm.cuda()
            self.activation.cuda()

    def forward(self, x):
        z = self.conv(x)
        z = self.batch_norm(z)

        return self.activation(z)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, device="cpu"):
        super().__init__()
        self.initial_block = nn.Sequential(OrderedDict([
            ('initial_conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        self.residual_block_1 = ResidualBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_block_2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if device == "cuda" and torch.cuda.is_available():
            self.initial_block.cuda()
            self.residual_block_1.cuda()
            self.residual_block_2.cuda()

    def forward(self, x):
        x = self.initial_block(x)
        z = self.residual_block_1(x)
        z = self.residual_block_2(z)

        return z + x


class ResNetEncoder(nn.Module):
    def __init__(self, blocks_in_channels, blocks_width, device="cpu"):
        super().__init__()
        self.stacks = [
            ResidualStack(in_channels=w_in, out_channels=w_out).to(device) for w_in, w_out in zip(blocks_in_channels, blocks_width)
        ]

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)

        return x


class StackMLP(nn.Module):
    def __init__(self, in_feats, out_feats, device):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=in_feats, out_features=2048).to(device)
        self.layer_2 = nn.Linear(in_features=2048, out_features=out_feats).to(device)
        self.activation = nn.ReLU().to(device)

    def forward(self, x):
        z = self.layer_1(x)

        return self.activation(self.layer_2(z))


class CausalIDMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, feats_dim, conv3d_in_channels, conv3d_out_channels, resnet_in_channels,
                 resnet_out_channels, input_size, use_conv3d=False, device="cpu"):
        assert device in ["cpu", "cuda"], "Unknown device."
        super().__init__(gym.spaces.box.Box(low=0.0, high=1.0, shape=input_size), features_dim=feats_dim)
        self.observation_space = observation_space
        self.use_conv3d = use_conv3d
        self.device = device
        if self.use_conv3d:
            self.conv_3d = nn.Conv3d(
                in_channels=conv3d_in_channels,
                out_channels=conv3d_out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0)
            )
            if device == "cuda" and torch.cuda.is_available():
                self.conv_3d.cuda()

        self.resnet = ResNetEncoder(
            blocks_in_channels=resnet_in_channels,
            blocks_width=resnet_out_channels,
            device=device
        )

        mlp_in_feats = self._compute_flatten_feats(input_size, device)
        self.mlp = StackMLP(in_feats=mlp_in_feats, out_feats=feats_dim, device=device)

        # Initialize weights
        self.conv_3d.apply(self._init_weights)
        self.resnet.apply(self._init_weights)
        self.mlp.apply(self._init_weights)


        if device == "cuda" and torch.cuda.is_available():
            self.resnet.cuda()
            self.mlp.cuda()

    def forward(self, x):
        if self.use_conv3d:
            z = self.conv_3d(x)
            z = self.resnet(z.squeeze(2))
        else:
            z = self.resnet(x.squeeze(1))

        return self.mlp(torch.flatten(z, start_dim=1))

    def _compute_flatten_feats(self, input_size, device):
        x = torch.rand(size=input_size).to(device)
        with torch.no_grad():
            if self.use_conv3d:
                x = self.conv_3d(x)
                x = self.resnet(x.squeeze(2))
            else:
                x = self.resnet(x.squeeze(0))

            x = torch.flatten(x)
            if device == "cuda":
                x = x.cpu().numpy()
            else:
                x = x.numpy()

        return int(x.shape[0])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.fill_(0.01)
