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


# ============================================================
# Projection Head (2-layer MLP with BatchNorm)
# ============================================================
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


# ============================================================
# SimCLR Transform
# ============================================================
class SimCLRTransform:
    def __init__(self, image_size=128):
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


# ============================================================
# Custom Flat-Folder Dataset
# ============================================================
class SimCLRDataset(Dataset):
    def __init__(self, root, image_size=128):
        self.image_paths = glob.glob(os.path.join(root, "*.jpg"))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")

        self.transform = SimCLRTransform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        x1, x2 = self.transform(img)
        return x1, x2


# ============================================================
# SimCLR ResNet Encoder + Projector
# ============================================================
class SimCLRResNet(nn.Module):
    def __init__(self, feature_dim=128, backbone="resnet50"):
        super().__init__()

        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # Remove final classifier
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder_out_dim = resnet.fc.in_features

        self.projector = ProjectionHead(
            in_dim=self.encoder_out_dim,
            hidden_dim=1024,
            out_dim=feature_dim
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = F.normalize(self.projector(h), dim=1)
        return h, z


# ============================================================
# NT-Xent Loss
# ============================================================
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask

    pos = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / exp_sim.sum(dim=1))
    return loss.mean()


# ============================================================
# Training Loop with Resume Support
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume")
    args = parser.parse_args()

    # ------------------ Load config ------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = config.get("fp16", False)

    # ------------------ Dataset & Loader ------------------
    dataset = SimCLRDataset(config["data_dir"], config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    # ------------------ Model ------------------
    model = SimCLRResNet(feature_dim=config["feature_dim"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ------------------ Resume Logic ------------------
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        print(f"Starting from epoch {start_epoch}")

    # Make directories
    os.makedirs(config["save_dir"], exist_ok=True)

    # ------------------ Training Loop ------------------
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0

        for x_i, x_j in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = nt_xent_loss(z_i, z_j, config["temperature"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")

        # ------------------ Save checkpoint ------------------
        ckpt_path = os.path.join(config["save_dir"], f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)

    # ------------------ Save Final Backbone ------------------
    backbone_path = os.path.join(config["save_dir"], config["save_backbone_name"])
    torch.save(model.encoder.state_dict(), backbone_path)
    print(f"Backbone saved at: {backbone_path}")


if __name__ == "__main__":
    main()
