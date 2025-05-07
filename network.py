# network.py
"""
Defines the neural network architecture (Policy-Value Network).
Based on the AlphaGo Zero paper, adapted for Chess.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import config


class ResidualBlock(nn.Module):
    """
    A single residual block as used in AlphaGo Zero.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying residual connection.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # skip connection
        out += identity
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    """
    The main neural network combining policy and value heads.
    """

    def __init__(self):
        super().__init__()
        # --- Convolutional Body ---
        self.conv_input = nn.Conv2d(
            config.INPUT_CHANNELS,
            config.CONV_FILTERS,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_input = nn.BatchNorm2d(config.CONV_FILTERS)

        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(config.CONV_FILTERS) for _ in range(config.RESIDUAL_BLOCKS)]
        )

        # --- Policy Head ---
        # Predicts move probabilities
        self.policy_conv = nn.Conv2d(config.CONV_FILTERS, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # Flatten and fully connected layer
        # Input features: 2 channels * BOARD_SIZE * BOARD_SIZE
        self.policy_fc = nn.Linear(
            2 * config.BOARD_SIZE * config.BOARD_SIZE, config.NUM_ACTIONS
        )
        # Output: Logits for each possible action

        # --- Value Head ---
        # Predicts the expected outcome of the game (-1 to 1)
        self.value_conv = nn.Conv2d(config.CONV_FILTERS, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        # Flatten and fully connected layers
        self.value_fc1 = nn.Linear(32 * config.BOARD_SIZE * config.BOARD_SIZE, 256)
        # Single output value
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor representing the board state(s).
               Shape: (batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

        Returns:
            A tuple containing:
            - policy_logits: Raw output scores for each action.
                             Shape: (batch_size, NUM_ACTIONS)
            - value: Predicted value of the board state(s).
                     Shape: (batch_size, 1)
        """
        # --- Body ---
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.residual_blocks(x)

        # --- Policy Head ---
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)
        # Note: Softmax is usually applied *after* this during MCTS or loss calculation

        # --- Value Head ---
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and 1

        return policy_logits, value
