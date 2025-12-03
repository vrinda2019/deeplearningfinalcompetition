import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm

########################################
# 1. SimCLR Dataset (no class folders)
########################################

class SimCLRDataset(Dataset):
    def __init__(self, root, image_size=224):
        self.root = root

        # Find all image files
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(exts)
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in {root}")

        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


########################################
# 2. SimCLR Model with ViT Encoder
########################################

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLRViT(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # vit_tiny_patch16_224 exists in timm
        self.encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.encoder.num_features
        self.projector = ProjectionHead(embed_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return h, z


########################################
# 3. NT-Xent Loss
########################################

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.mm(z, z.t()) / temperature
    mask = ~torch.eye(2 * N, dtype=bool).to(z.device)
    sim = sim[mask].view(2 * N, 2 * N - 1)

    positives = torch.sum(z1 * z2, dim=-1) / temperature
    positives = torch.cat([positives, positives], dim=0)

    labels = torch.arange(2 * N).to(z.device)

    return F.cross_entropy(sim, labels)


########################################
# 4. Training Loop
########################################

def main():
    data_dir = "/scratch/vt2370/dataset/cc3m_all/train"
    image_size = 224
    batch_size = 256
    epochs = 100
    lr = 3e-4

    print("Loading dataset...")
    dataset = SimCLRDataset(data_dir, image_size)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8,
                        drop_last=True)

    print("Building model...")
    model = SimCLRViT().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting SSL pretraining...")
    for epoch in range(epochs):
        for x1, x2 in loader:
            x1, x2 = x1.cuda(), x2.cuda()
            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    print("Training finished!")


if __name__ == "__main__":
    main()

