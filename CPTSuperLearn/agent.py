import os
from typing import List
import random
import numpy as np
from collections import deque, namedtuple
import torch
from torch import nn
from torch import optim


class DQLAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 64, nb_steps_update: int = 10, model_path: str = None):
        """
        Initialize the DQL agent

        Parameters:
        -----------
        :param state_size: size of the state
        :param action_size: size of the action
        :param learning_rate: learning rate
        :param gamma: discount factor
        :param epsilon_start: starting epsilon
        :param epsilon_end: ending epsilon
        :param epsilon_decay: epsilon decay
        :param memory_size: size of the replay buffer
        :param batch_size: batch size
        :param nb_steps_update: number of steps to update target network
        :param model_path: path to save the model
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.nb_steps_update = nb_steps_update
        self.nb_step = 0
        self.model_path = model_path

    def save_model(self, path: str):
        """
        Save the model

        Parameters:
        -----------
        :param path: path to save the model
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model_path = path
        torch.save(self.qnetwork_local.state_dict(), path)

    def load_model(self):
        """
        Load the model

        Parameters:
        -----------
        :param path: path to load the model
        """

        if self.model_path is None:
            raise ValueError("Model has not been saved yet")

        self.qnetwork_local.load_state_dict(torch.load(self.model_path,  weights_only=True))


    def get_next_action(self, state: torch.Tensor, training: bool = True) -> int:
        r"""
        Get the next action

        Parameters:
        -----------
        :param state: current state
        :param training: training mode
        :return: next action
        """
        # if training
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.qnetwork_local.network[-1].out_features - 1)
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return action_values.argmax().item()

    def step(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, terminal: bool):
        """
        Take a step with the agent.
        Add the experience to the replay buffer and learn from it.

        Parameters:
        -----------
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state
        :param terminal: terminal status
        """
        self.memory.push(state, action, reward, next_state, terminal)

        # update the target network with the local network
        self.nb_step = (self.nb_step + 1) % self.nb_steps_update
        if self.nb_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # check if memory buffer has more experiences than batch_size
        # training only occurs when this condition meets
        if len(self.memory) > self.batch_size:
            self._learn()

        # update epsilon when terminal state is reached
        if terminal:
            self._update_epsilon()

    def _learn(self):
        """
        Learn from the replay buffer
        """

        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        # Convert to tensors and move to device
        states = torch.stack(batch.state).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float).to(self.device)

        # Get Q values
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss and update
        loss = nn.MSELoss()(Q_expected, Q_targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_epsilon(self):
        """
        Update epsilon
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()

        # Improved architecture with batch normalization
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters:
        -----------
        :param state: current state
        :return: action values
        """
        return self.network(state)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer

        Parameters:
        -----------
        :param capacity: size of the replay buffer
        """

        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                   ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        """
        Push a transition into the replay buffer

        Parameters:
        -----------
        :param args: transition
        """
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size: int) -> List[torch.tensor]:
        """
        Sample a batch of transitions

        Parameters:
        -----------
        :param batch_size: size of the batch
        :return: batch of transitions
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        Return the length of the replay buffer

        Returns:
        --------
        :return: length of the replay buffer
        """
        return len(self.buffer)