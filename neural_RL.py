import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda


# torch.manual_seed(14)
# np.random.seed(14)


class CustomDataset(Dataset):
    """
    Custom dataset class for pytorch
    """
    def __init__(self, data: np.array, labels: np.array, transform=None, target_transform=None):
        """
        Initialize the dataset

        Parameters:
        ----------
        data (np.array): data
        labels (np.array): labels
        transform (): transform to apply to the data
        target_transform (): transform to apply to the labels
        """
        self.features = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Return the data and label at the given index

        Parameters:
        ----------
        idx (int): index of the data and label to return
        """
        feat = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feat = self.transform(feat)
        if self.target_transform:
            label = self.target_transform(label)
        return feat, label


class SimpleNeuralNetwork(nn.Module):
    """
    Simple neural network class based on pytorch
    """
    def __init__(self, size_input: int, size_output: int, hidden_layers: list, lr: float = 0.001) -> None:
        """
        Initialize the neural network

        Parameters:
        ----------
        size_input (int): size of the input layer
        size_output (int): size of the output layer
        hidden_layers (list): list of the size of each hidden layer
        lr (float): learning rate
        """
        super().__init__()

        self.hidden_layers = []
        for i, _ in enumerate(hidden_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(size_input, hidden_layers[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        self.output = nn.Linear(hidden_layers[-1], size_output)

        # activation function
        self.activation_fct = F.relu
        # loss function
        self.criterion = nn.MSELoss()
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # variables
        self.size_input = size_input
        self.size_output = size_output

        self.loss = []
        self.prediction = []

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Parameters:
        ----------
        x (torch.Tensor): input tensor
        """
        for hidden in self.hidden_layers:
            x = self.activation_fct(hidden(x))
        x = self.output(x)
        return x

    def run(self, output: torch.Tensor, target: torch.Tensor):
        """
        Run the neural network

        Parameters:
        ----------
        output (torch.Tensor): expected output
        target (torch.Tensor): target output
        """
        if torch.cuda.is_available():
            output = output.to("cuda")
            target = target.to("cuda")
            
        loss = self.criterion(output, target.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item())


    def predict(self, data: torch.Tensor):
        """
        Predict the output of the neural network

        Parameters:
        ----------
        data (torch.Tensor): input tensor to predict
        """
        # do not update the weights
        with torch.no_grad():
            self.prediction = self.forward(data)

    def save_model(self, destination: str, file_name: str):
        """
        Save the pytorch model

        Parameters:
        ----------
        destination (str): destination folder
        file_name (str): file name
        """

        if not os.path.exists(destination):
            os.makedirs(destination)

        checkpoint = {'input_size': self.size_input,
                      'output_size': self.size_output,
                      'hidden_layers': [each.out_features for each in self.hidden_layers],
                      'state_dict': self.state_dict()}

        torch.save(checkpoint, os.path.join(destination, file_name))

    @staticmethod
    def load_model(file_path: str):
        """
        Load the pytorch model

        Parameters:
        ----------
        file_path (str): path to the model
        """

        checkpoint = torch.load(file_path)
        model = SimpleNeuralNetwork(checkpoint['input_size'],
                                    checkpoint['output_size'],
                                    checkpoint['hidden_layers'])
        model.load_state_dict(checkpoint['state_dict'])

        return model
