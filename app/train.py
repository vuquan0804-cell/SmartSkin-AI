from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "model"

DEFAULT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class HamDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], image_size: int) -> None:
        self.items = items
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label = self.items[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def resolve_image_path(image_id: str) -> Path:
    filename = f"{image_id}.jpg"
    part1 = DATA_DIR / "HAM10000_images_part_1" / filename
    if part1.exists():
        return part1
    part2 = DATA_DIR / "HAM10000_images_part_2" / filename
    if part2.exists():
        return part2
    raise FileNotFoundError(f"Image not found: {image_id}")


def build_items(metadata_path: Path, labels: List[str]) -> List[Tuple[Path, int]]:
    df = pd.read_csv(metadata_path)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    items: List[Tuple[Path, int]] = []
    for _, row in df.iterrows():
        image_path = resolve_image_path(row["image_id"])
        label = label_to_idx[row["dx"]]
        items.append((image_path, label))
    return items


def train(args: argparse.Namespace) -> None:
    metadata_path = DATA_DIR / "HAM10000_metadata.csv"
    labels = DEFAULT_LABELS
    items = build_items(metadata_path, labels)

    y = [label for _, label in items]
    train_items, val_items = train_test_split(
        items, test_size=args.val_split, stratify=y, random_state=42
    )

    train_dataset = HamDataset(train_items, args.image_size)
    val_dataset = HamDataset(val_items, args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(labels))
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / max(total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss: {running_loss / len(train_dataset):.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump({"labels": labels}, handle, indent=2)

    print(f"Training complete. Best val acc: {best_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
