import cv2
import numpy as np
import time
import argparse
import torch
import torchvision
from model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--weight', default='fashion-mnist.pth', type=str)
parser.add_argument('--image', default='test.png', type=str)
args = parser.parse_args()

start = time.time()
# Load Weights
model = MyModel()
model.load_state_dict(torch.load(args.weight))
model.eval()

img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

device = torch.device(args.device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0, std=1)])
tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()
output = np.argmax(preds)
end = time.time()
print(f"prediction result: {output} , inference time equal: {end-start}")




'''
# Inference
import cv2
import numpy as np

model.train(False) # model.eval()

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()
output = np.argmax(preds)
output
'''