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
# Data Augmentations
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
                                 [0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

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
# Projection Head
# ----------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# ----------------------
# ViT SSL Model
# ----------------------
class SimCLRViT(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0   # no classification head, return embeddings
        )

        embed_dim = self.encoder.num_features
        self.projector = ProjectionHead(embed_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return h, z

# ----------------------
# NT-Xent Loss
# ----------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)
    mask = ~torch.eye(2*batch_size, dtype=bool, device=z.device)

    sim_exp = torch.exp(similarity_matrix / temperature) * mask

    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(pos_sim / sim_exp.sum(dim=1))
    return loss.mean()

# ----------------------
# Training Loop
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Seeds
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["save_dir"], exist_ok=True)

    # Dataset
    dataset = SimCLRDataset(config["data_dir"], config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )

    # Model
    model = SimCLRViT(feature_dim=config["feature_dim"]).to(device)
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Train
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for x_i, x_j in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature=config["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")

    # Save encoder weights only
    torch.save(
        model.encoder.state_dict(),
        os.path.join(config["save_dir"], config["save_backbone_name"])
    )

    print("Finished training. Backbone saved.")

if __name__ == "__main__":
    main()
