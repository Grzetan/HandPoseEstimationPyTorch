import torch
from torchvision.transforms import Compose
from model import HandPoseEstimator
import model2
from architecture import architecture
from PanopticHandDataset import PanopticHandDataset
from FreiHandDataset import FreiHandDataset
from transforms import *
import matplotlib.pyplot as plt

device = torch.device('cpu')
checkpoint = torch.load('./model-1644740665.669266-epoch0.pth', map_location=device)

model = HandPoseEstimator(architecture)
# config = {
#   "name": "ARB",
#   "dataset":"HandDataset",
#   "data_root":"data_sample/cmuhand",
#   "ARB": "ARB_Cat",
#   "batch_size": 4,
#   "learning_rate": 1e-4,
#   "epochs": 15
# }

# model = model2.light_Model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transforms = Compose([
    Resize(img_size=224),
    # RandomPadding(),
    # RandomVerticalFlip(0.5),
    # RandomColorJitter(),
    ToTensor()
])

raw_transforms = Compose([
    Resize(),
    # RandomPadding(),
    # RandomVerticalFlip(0.5),
    # RandomColorJitter()
])

# dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=transforms)
# raw_dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=raw_transforms)
dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth', transforms=transforms)
raw_dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth')

loader = dataset.get_loader(batch_size=1)

for i, (img, points) in enumerate(loader):
    img = img.to(device)
    pred_points = model(img)[0]
    print(pred_points)
    pred_points = pred_points.detach().squeeze().numpy()
    pred_points *= 224
    org_img, _ = raw_dataset[i]
    fig, ax = plt.subplots()
    ax.imshow(org_img)
    for p in pred_points:
        ax.scatter(p[0], p[1], c='r', s=10)
    plt.show()