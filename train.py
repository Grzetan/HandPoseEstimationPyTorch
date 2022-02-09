import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
from model import HandPoseEstimator
from architecture import architecture
from PanopticHandDataset import PanopticHandDataset
from FreiHandDataset import FreiHandDataset
from transforms import *
import time
import model2

def train(model, criterion, optimizer, loader, scheduler, epochs=1, save=300):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(epochs):
        losses = []
        start = time.time()
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
            print(f'\rEpoch {epoch+1}/{epochs}, Progress: {round(i/len(loader) * 100, 3)}%, Mean Loss: {round(sum(losses) / len(losses), 5)}, Loss: {round(loss.item(), 5)}, Saved {i//save} times, Time Elapsed: {(time.time() - start) // 60}min', end='')
            if i % save == 0:
                torch.save(model.state_dict(), f'./model-epoch{epoch}-loss{round(sum(losses) / len(losses), 5)}.pth')
        torch.save(model.state_dict(), f'./model-epoch{epoch}-loss{round(sum(losses) / len(losses), 5)}.pth')
        print(" ")

transforms = Compose([
    # Resize(),
    ToTensor()
])

model = HandPoseEstimator(architecture)

# configs = {
#   "name": "ARB",
#   "dataset":"HandDataset",
#   "data_root":"data_sample/cmuhand",
#   "ARB": "ARB_Cat",
#   "batch_size": 4,
#   "learning_rate": 1e-4,
#   "epochs": 15
# }
# model = model.light_Model(configs)

criterion = nn.MSELoss(reduction='sum')
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=transforms)
# dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth', transforms=transforms)

loader = dataset.get_loader(batch_size=2)
# step_size = len(loader) * 8
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=step_size, mode='triangular2')
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(loader), epochs=15)

train(model, criterion, optimizer, loader, scheduler, epochs=15)