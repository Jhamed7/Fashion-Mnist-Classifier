import time
import argparse
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt

from model import MyModel
from test import test_model
from utils import calc_accuracy
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
args = parser.parse_args()

wandb.login()
wandb.init(project="Fashion-Mnist pytorch")

model = MyModel()
device = torch.device(args.device)
model = model.to(device)
model.train(True)

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr

# Data preparation
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0, std=1)])
dataset_train = torchvision.datasets.FashionMNIST(root='datasets', train=True, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Compile
# optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()


# Training
total_loss = []
total_acc = []
for epoch in (range(epochs)):
    train_loss = 0
    train_acc = 0
    
    for images, labels in (tqdm.tqdm(train_loader)):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        preds = model(images)

        loss = loss_function(preds, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss
        train_acc += calc_accuracy(preds, labels)
      
    total_loss.append(train_loss / len(train_loader))
    total_acc.append(train_acc / len(train_loader))

    wandb.log({'accuracy': total_acc[-1], 'loss': total_loss[-1]})
    
    print(f"Epoch: {epoch}, loss: {total_loss[-1]}, Accuracy: {total_acc[-1]}")
    test_model(model, batch_size, device)

plt.plot(total_loss,'g*',label="loss")
plt.plot(total_acc, 'ro', label="accuracy")
plt.legend(loc="upper right")
plt.show()

# Save Model
torch.save(model.state_dict(), "fashion-mnist.pth")



