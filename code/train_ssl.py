import os
import random
import argparse
import glob
from PIL import Image
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------------------
# Projection Head (wider, with BatchNorm)
# ----------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# SimCLR Transform
# ----------------------
class SimCLRTransform:
    def __init__(self, image_size=128):
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
# Custom Dataset for Flat Folder
# ----------------------
class SimCLRDataset(Dataset):
    def __init__(self, root, image_size=128):
        self.image_paths = glob.glob(os.path.join(root, "*.jpg"))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = SimCLRTransform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        x_i, x_j = self.transform(img)
        return x_i, x_j

# ----------------------
# SimCLR ResNet + Projector
# ----------------------
class SimCLRResNet(nn.Module):
    def __init__(self, feature_dim=128, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        modules = list(resnet.children())[:-1]  # remove classifier
        self.encoder = nn.Sequential(*modules)
        self.encoder_out_dim = resnet.fc.in_features

        self.projector = ProjectionHead(self.encoder_out_dim, hidden_dim=1024, out_dim=feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = F.normalize(self.projector(h), dim=1)
        return h, z

# ----------------------
# NT-Xent Loss
# ----------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).to(z.device)
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
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    # Device and FP16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = config.get("fp16", False)

    os.makedirs(config["save_dir"], exist_ok=True)

    # Dataset & DataLoader
    dataset = SimCLRDataset(config["data_dir"], config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"]
    )

    # Model
    model = SimCLRResNet(feature_dim=config["feature_dim"]).to(device)
    if torch.cuda.device_count() > 1 and config.get("multi_gpu", False):
        model = nn.DataParallel(model)

    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Dataset size: {len(dataset)} images")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Training
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for x_i, x_j in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                h_i, z_i = model(x_i)
                h_j, z_j = model(x_j)
                loss = nt_xent_loss(z_i, z_j, temperature=config["temperature"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(config["save_dir"], f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)

    # Save final backbone
    backbone_path = os.path.join(config["save_dir"], config["save_backbone_name"])
    torch.save(model.encoder.state_dict(), backbone_path)
    print(f"Backbone saved to {backbone_path}")

if __name__ == "__main__":
    main()
