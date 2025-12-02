import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import torch.nn.functional as F
import os
import random

# ----------------------
# Dual Augmentation Transform
# ----------------------
class SimCLRTransform:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        return self.transform(x), self.transform(x)  # two augmented views

# ----------------------
# Dataset Wrapper
# ----------------------
class SimCLRDataset(datasets.ImageFolder):
    def __init__(self, root, image_size):
        super().__init__(root=root)
        self.simclr_transform = SimCLRTransform(image_size)
    
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        x_i, x_j = self.simclr_transform(img)
        return x_i, x_j

# ----------------------
# Vision Transformer Backbone
# ----------------------
class ViT_SSL(nn.Module):
    def __init__(self, feature_dim=128, backbone_name="vit_tiny_patch16_96"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=feature_dim
        )
    
    def forward(self, x):
        return self.backbone(x)

# ----------------------
# NT-Xent Loss
# ----------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    batch_size = z_i.size(0)
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = -torch.log(pos_sim / exp_sim.sum(dim=1))
    return loss.mean()

# ----------------------
# Training Loop
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config["save_dir"], exist_ok=True)

    # Dataset & Loader
    dataset = SimCLRDataset(config["data_dir"], config["image_size"])
    loader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        drop_last=True,
                        num_workers=config["num_workers"])

    # Model
    model = ViT_SSL(feature_dim=config["feature_dim"], backbone_name=config["backbone"]).to(device)

    # Print trainable params and image resolution
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params/1e6:.2f}M")
    print(f"Image Resolution: {config['image_size']}x{config['image_size']}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for x_i, x_j in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j = model(x_i), model(x_j)
            loss = nt_xent_loss(z_i, z_j, temperature=config["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(loader):.4f}")

    # Save backbone
    torch.save(model.backbone.state_dict(), os.path.join(config["save_dir"], config["save_backbone_name"]))
    print(f"Backbone saved to {config['save_dir']}/{config['save_backbone_name']}")

if __name__ == '__main__':
    main()

