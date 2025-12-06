"""
Create Kaggle Submission with KNN or Linear Probe on SimCLR features
===================================================================

Usage:
    python knn.py \
        --data_dir ./kaggle_data \
        --checkpoint checkpoint_epoch_100.pth \
        --use_linear_probe \
        --output submission.csv \
        --batch_size 32 \
        --k 5
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.simclr import SimCLRResNet  # Your SimCLR encoder
from scripts.eval import extract_features  # Vectorized feature extractor

# ---------------------------
# Dataset
# ---------------------------
class ImageDataset(Dataset):
    def __init__(self, image_dir, filenames, labels=None, resolution=128):
        self.image_dir = Path(image_dir)
        self.filenames = filenames
        self.labels = labels
        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.labels is not None:
            return img, int(self.labels[idx]), self.filenames[idx]
        return img, self.filenames[idx]

def collate_fn(batch):
    sample0 = batch[0]
    if len(sample0) == 3:
        imgs, labels, fnames = [], [], []
        for img, lab, fname in batch:
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            imgs.append(img)
            labels.append(lab)
            fnames.append(fname)
        return torch.stack(imgs, 0), torch.tensor(labels), fnames
    else:
        imgs, fnames = [], []
        for img, fname in batch:
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            imgs.append(img)
            fnames.append(fname)
        return torch.stack(imgs, 0), fnames

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features_from_loader(model, loader, device):
    model.eval()
    all_feats = []
    all_labels = []
    all_fnames = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels, fnames = batch
                all_labels.extend(labels.numpy().tolist())
            else:
                imgs, fnames = batch
            imgs = imgs.to(device)
            feats = extract_features(model, imgs, pool="mean")  # Vectorized
            feats = feats.cpu().numpy()
            # Normalize features
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            all_feats.append(feats)
            all_fnames.extend(fnames)

    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats, all_labels, all_fnames

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_linear_probe', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)

    # Load CSVs
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Datasets
    train_ds = ImageDataset(data_dir/'train', train_df['filename'], train_df['class_id'])
    val_ds   = ImageDataset(data_dir/'val', val_df['filename'], val_df['class_id'])
    test_ds  = ImageDataset(data_dir/'test', test_df['filename'], labels=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Load SimCLR encoder
    model = SimCLRResNet(feature_dim=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print("SimCLR encoder loaded successfully.")

    # Extract features
    train_feats, train_labels, _ = extract_features_from_loader(model, train_loader, device)
    val_feats, val_labels, _     = extract_features_from_loader(model, val_loader, device)
    test_feats, _, test_fnames    = extract_features_from_loader(model, test_loader, device)

    # Train classifier
    if args.use_linear_probe:
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial', n_jobs=-1)
        clf.fit(train_feats, train_labels)
        train_acc = clf.score(train_feats, train_labels)
        val_acc = clf.score(val_feats, val_labels)
        print(f"Linear Probe -> Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
    else:
        clf = KNeighborsClassifier(n_neighbors=args.k, metric='cosine', weights='distance', n_jobs=-1)
        clf.fit(train_feats, train_labels)
        train_acc = clf.score(train_feats, train_labels)
        val_acc = clf.score(val_feats, val_labels)
        print(f"KNN -> Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Predictions
    preds = clf.predict(test_feats)
    submission = pd.DataFrame({'id': test_fnames, 'class_id': preds})
    submission.to_csv(args.output, index=False)
    print(f"Submission saved: {args.output}")

if __name__ == "__main__":
    main()
