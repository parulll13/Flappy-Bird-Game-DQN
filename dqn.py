import torch
import torch.nn as nn

# Define Deep Q-Network class
class DQN(nn.Module):

    # Initialize input size, output size and hidden layer size
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):

        # Call parent class constructor
        super(DQN, self).__init__()

        # Create neural network architecture
        self.model = nn.Sequential(

            # Input layer → Hidden layer
            nn.Linear(state_dim, hidden_dim),

            # Add non-linearity
            nn.ReLU(),

            # Hidden layer → Output layer
            nn.Linear(hidden_dim, action_dim)
        )

    # Forward pass: send input through network
    def forward(self, x):
        return self.model(x)