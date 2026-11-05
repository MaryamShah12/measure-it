"""Train a U-Net to segment the blue notebook.

Model: U-Net with a pretrained ResNet18 encoder, 1 class.
Input: 256 by 256. Loss: Dice plus binary cross entropy.
Split: read from dataset/splits.txt, fixed before training.
Test images are never used for training, tuning, or checkpoint choice.
The checkpoint saved is the one with the best validation IoU.

Run from the repo root, in Colab with a T4 GPU:
    pip install -q segmentation-models-pytorch
    python models/train.py

Outputs: models/best_model.pth, models/training_history.json,
models/test_metrics.json, models/versions.json, models/test_overlays/
"""
import argparse
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split(path):
    splits = {"train": [], "val": [], "test": []}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                splits[parts[0]].append(parts[1])
    return splits


class NotebookDataset(Dataset):
    def __init__(self, names, root, size, augment=False):
        self.names = names
        self.root = root
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.root, "dataset/images", name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.root, "dataset/masks", name.replace(".jpg", ".png")),
            cv2.IMREAD_GRAYSCALE,
        )
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            angle = random.uniform(-30, 30)
            center = (self.size / 2, self.size / 2)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, m, (self.size, self.size))
            mask = cv2.warpAffine(mask, m, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        mask = (mask > 127).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, name


def iou_score(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    inter = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return inter / union if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("torch:", torch.__version__)
    print("segmentation_models_pytorch:", smp.__version__)

    splits = read_split(os.path.join(args.root, "dataset/splits.txt"))
    print("split sizes:", {k: len(v) for k, v in splits.items()})

    train_ds = NotebookDataset(splits["train"], args.root, args.size, augment=True)
    val_ds = NotebookDataset(splits["val"], args.root, args.size, augment=False)
    test_ds = NotebookDataset(splits["test"], args.root, args.size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)

    best_val_iou = -1.0
    history = []
    os.makedirs(os.path.join(args.root, "models"), exist_ok=True)
    best_path = os.path.join(args.root, "models/best_model.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = dice(out, masks) + bce(out, masks)
            loss.backward()
            opt.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_ious = []
        with torch.no_grad():
            for imgs, masks, _ in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss += (dice(out, masks) + bce(out, masks)).item() * imgs.size(0)
                probs = torch.sigmoid(out)
                for i in range(imgs.size(0)):
                    val_ious.append(iou_score(probs[i].cpu(), masks[i].cpu()))
        val_loss /= len(val_ds)
        val_iou = float(np.mean(val_ious))
        sched.step(val_loss)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou})
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print("saved best model, val_iou=", round(val_iou, 4))

    with open(os.path.join(args.root, "models/training_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(args.root, "models/versions.json"), "w") as f:
        json.dump(
            {
                "torch": torch.__version__,
                "segmentation_models_pytorch": smp.__version__,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch": args.batch,
                "size": args.size,
                "lr": args.lr,
                "encoder": "resnet18",
                "device": device,
            },
            f,
            indent=1,
        )

    # Test only once, with the best checkpoint, never used for tuning.
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    per_image = []
    overlay_dir = os.path.join(args.root, "models/test_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    with torch.no_grad():
        for imgs, masks, names in test_dl:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs))[0, 0].cpu().numpy()
            pred = (probs > 0.5).astype(np.uint8)
            true = masks[0, 0].numpy()
            inter = np.logical_and(pred > 0, true > 0).sum()
            union = np.logical_or(pred > 0, true > 0).sum()
            iou = float(inter / union) if union else 1.0
            tp = inter
            fp = np.logical_and(pred > 0, true == 0).sum()
            fn = np.logical_and(pred == 0, true > 0).sum()
            precision = float(tp / (tp + fp)) if tp + fp else 1.0
            recall = float(tp / (tp + fn)) if tp + fn else 1.0
            per_image.append({"image": names[0], "iou": iou, "precision": precision, "recall": recall})
            # Overlay: green is true mask edge, red is predicted mask edge.
            orig = cv2.imread(os.path.join(args.root, "dataset/images", names[0]))
            orig = cv2.resize(orig, (args.size, args.size))
            true_u8 = (true * 255).astype(np.uint8)
            pred_u8 = (pred * 255).astype(np.uint8)
            tc, _ = cv2.findContours(true_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pc, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(orig, tc, -1, (0, 255, 0), 2)
            cv2.drawContours(orig, pc, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(overlay_dir, names[0]), orig)
            print(names[0], "iou=", round(iou, 4), "precision=", round(precision, 4), "recall=", round(recall, 4))

    mean_iou = float(np.mean([p["iou"] for p in per_image]))
    print("TEST mean IoU:", round(mean_iou, 4))
    with open(os.path.join(args.root, "models/test_metrics.json"), "w") as f:
        json.dump({"mean_iou": mean_iou, "per_image": per_image}, f, indent=1)


if __name__ == "__main__":
    main()
