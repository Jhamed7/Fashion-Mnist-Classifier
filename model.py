import torch

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # input shape batch*28*28
        x = x.reshape((x.shape[0], 784))

        x = self.fc1(x)
        x = torch.relu(x)
        # x = torch.dropout(x, 0.2, train=True)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x