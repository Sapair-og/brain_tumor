"""Leak-free data pipeline for the Kaggle Brain Tumor MRI dataset.

Expected layout (the original Kaggle download, NOT merged):
    <root>/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    <root>/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

The official Testing split is kept as the held-out test set. Exact-duplicate images
(by MD5) are removed inside each split and any training image that also appears in
Testing is dropped, so test accuracy is not inflated by leakage.
"""
import hashlib
import os
from collections import Counter

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .model import CLASSES

IMG_EXTS = (".png", ".jpg", ".jpeg")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(img_size=224):
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(img_size=224):
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _scan(split_dir):
    """Return deduplicated [(path, label)] and the set of hashes in this split."""
    items, seen = [], set()
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        for name in sorted(os.listdir(cls_dir)):
            if not name.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(cls_dir, name)
            h = _md5(path)
            if h in seen:
                continue
            seen.add(h)
            items.append((path, label, h))
    return items, seen


class MRIDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i][:2]
        return self.transform(Image.open(path).convert("RGB")), label


def make_splits(root, val_size=0.15, seed=42):
    train_items, _ = _scan(os.path.join(root, "Training"))
    test_items, test_hashes = _scan(os.path.join(root, "Testing"))
    leaked = sum(h in test_hashes for _, _, h in train_items)
    train_items = [it for it in train_items if it[2] not in test_hashes]
    train_items, val_items = train_test_split(
        train_items, test_size=val_size, random_state=seed,
        stratify=[it[1] for it in train_items])
    print(f"Removed {leaked} train images duplicated in Testing.")
    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        counts = Counter(CLASSES[it[1]] for it in items)
        print(f"{name:5s} {len(items):5d} {dict(counts)}")
    return train_items, val_items, test_items


def make_loaders(root, img_size=224, batch_size=32, workers=4, seed=42):
    train_items, val_items, test_items = make_splits(root, seed=seed)
    # Balanced sampling so the minority class is seen as often as the majority.
    counts = Counter(it[1] for it in train_items)
    weights = [1.0 / counts[it[1]] for it in train_items]
    sampler = WeightedRandomSampler(weights, len(weights), generator=torch.Generator().manual_seed(seed))
    pin = torch.cuda.is_available()
    train = DataLoader(MRIDataset(train_items, train_transform(img_size)), batch_size,
                       sampler=sampler, num_workers=workers, pin_memory=pin, drop_last=True)
    val = DataLoader(MRIDataset(val_items, eval_transform(img_size)), batch_size,
                     num_workers=workers, pin_memory=pin)
    test = DataLoader(MRIDataset(test_items, eval_transform(img_size)), batch_size,
                      num_workers=workers, pin_memory=pin)
    return train, val, test
