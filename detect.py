import torch
from torchvision.transforms import Compose
from model import HandPoseEstimator
from architecture import architecture
from PanopticHandDataset import PanopticHandDataset
from transforms import *
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = HandPoseEstimator(architecture)
model.load_state_dict(torch.load('./model1.pth'))
model.eval()
model.to(device)

transforms = Compose([
    Resize(),
    ToTensor()
])

# dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=transforms)
# raw_dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json')
dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth', transforms=transforms)
raw_dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth')

loader = dataset.get_loader(batch_size=1)

for i, (img, points) in enumerate(loader):
    img = img.to(device)
    pred_points = model(img)
    print(pred_points)
    pred_points = pred_points.detach().squeeze().cpu().numpy()
    pred_points *= 224
    org_img, _ = raw_dataset[i]
    fig, ax = plt.subplots()
    ax.imshow(org_img)
    for p in pred_points:
        ax.scatter(p[0], p[1], c='r', s=10)
    plt.show()