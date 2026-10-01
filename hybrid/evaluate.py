"""Metrics, confusion matrix, latency and ensemble evaluation.

    python -m hybrid.evaluate --data path/to/dataset --ckpts checkpoints/brainhybrid.pt checkpoints/fastvit.pt
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .model import CLASSES, Ensemble, count_params, load_checkpoint


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        # Ensemble already returns probabilities; single models return logits.
        probs.append((out if isinstance(model, Ensemble) else out.softmax(-1)).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def evaluate(model, loader, device, plot_path=None):
    probs, labels = predict_proba(model, loader, device)
    preds = probs.argmax(1)
    if plot_path:
        save_confusion(labels, preds, plot_path)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "confusion": confusion_matrix(labels, preds, labels=range(len(CLASSES))).tolist(),
        "report": classification_report(labels, preds, labels=range(len(CLASSES)),
                                        target_names=CLASSES, digits=4, zero_division=0),
    }


def save_confusion(labels, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def measure_latency(model, img_size=224, runs=30):
    """Median single-image CPU latency in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    from .data import make_loaders

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_checkpoint(c, device)[0] for c in args.ckpts]
    model = models[0] if len(models) == 1 else Ensemble(models)
    _, _, test_dl = make_loaders(args.data, batch_size=args.batch_size, workers=args.workers)
    res = evaluate(model, test_dl, device, plot_path="checkpoints/ensemble_confusion.png")
    print(res["report"])
    print(f"acc {res['accuracy']:.4f} macro-F1 {res['macro_f1']:.4f} | "
          f"{count_params(model) / 1e6:.2f}M params | "
          f"{measure_latency(model.cpu()):.1f} ms/img CPU")


if __name__ == "__main__":
    main()
