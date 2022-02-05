import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
from model import HandPoseEstimator
from architecture import architecture
from dataset import FreiHandDataset
from transforms import *

def train(model, criterion, optimizer, loader, scheduler, epochs=1, save=300):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.half()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        losses = []

        for i, (imgs, points) in enumerate(loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            points = points.to(device)
            output = model(imgs)
            loss = torch.sqrt(criterion(output, points))
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            print(f'\rEpoch {epoch+1}/{epochs}, Progress: {round(i/len(loader) * 100, 3)}%, Loss: {round(sum(losses) / len(losses), 5)}, Saved {i//save} times', end='')
            if i % save == 0:
                torch.save(model.state_dict(), './model1.pth')
        torch.save(model.state_dict(), './model1.pth')

transforms = Compose([
    ToTensor()
])

model = HandPoseEstimator(architecture)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.0001)
dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=transforms)
loader = dataset.get_loader(batch_size=2)
step_size = len(loader) * 8
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, mode='triangular2')
train(model, criterion, optimizer, loader, scheduler)