"""Train a hybrid CNN+Transformer model.

    python -m hybrid.train --data path/to/brain-tumor-mri-dataset --arch brainhybrid
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from .data import make_loaders
from .evaluate import evaluate, measure_latency
from .model import ARCHS, CLASSES, build_model, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Folder containing Training/ and Testing/")
    p.add_argument("--arch", default="brainhybrid", choices=ARCHS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_with_warmup(optimizer, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"{args.arch}.pt")

    train_dl, val_dl, test_dl = make_loaders(args.data, args.img_size, args.batch_size,
                                             args.workers, args.seed)
    model = build_model(args.arch, len(CLASSES), pretrained=True, img_size=args.img_size).to(device)
    print(f"{args.arch}: {count_params(model) / 1e6:.2f}M params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_dl)
    scheduler = cosine_with_warmup(optimizer, warmup=len(train_dl), total=steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        val = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": total_loss / n, **{f"val_{k}": v for k, v in val.items() if k != "report"}})
        print(f"epoch {epoch:2d} loss {total_loss / n:.4f} val_acc {val['accuracy']:.4f} "
              f"val_f1 {val['macro_f1']:.4f} ({time.time() - t0:.0f}s)")

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save({"arch": args.arch, "classes": CLASSES, "img_size": args.img_size,
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    test = evaluate(model, test_dl, device, plot_path=os.path.join(args.out, f"{args.arch}_confusion.png"))
    results = {
        "arch": args.arch,
        "params_m": round(count_params(model) / 1e6, 2),
        "cpu_latency_ms": round(measure_latency(model.cpu(), args.img_size), 1),
        "test": test,
        "history": history,
    }
    with open(os.path.join(args.out, f"{args.arch}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(test["report"])
    print(f"TEST acc {test['accuracy']:.4f} macro-F1 {test['macro_f1']:.4f} "
          f"| {results['params_m']}M params | {results['cpu_latency_ms']} ms/img CPU")


if __name__ == "__main__":
    main()
