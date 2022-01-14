import torch
import torchvision
import argparse
from utils import calc_accuracy
from model import MyModel

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--weight', default='fashion-mnist.pth', type=str) 
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    return args

@torch.no_grad()
def test_model(model, batch_size, device):

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0, std=1)])
    dataset_test = torchvision.datasets.FashionMNIST(root='datasets', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    loss_function = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    test_acc = 0.0
    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = loss_function(preds, labels)

        test_loss += loss
        test_acc += calc_accuracy(preds, labels)
    total_loss = test_loss / len(test_loader)
    total_acc = test_acc / len(test_loader)
    
    print(f"\nloss_test: {total_loss}, Accuracy_test: {total_acc}")


if __name__ == "__main__":
    args = parse_opt()

    test_model = MyModel()
    test_model.load_state_dict(torch.load(args.weight))
    test_model.eval()

    test_model(model=test_model, batch_size=args.batch_size, device=args.device)