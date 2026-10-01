"""Gradio app for the hybrid CNN+Transformer brain tumor classifier.

Loads every checkpoint in ./checkpoints/*.pt (or the paths in $MODEL_CKPTS, comma-separated)
and averages their probabilities. Accepts any number of images; Grad-CAM overlays are shown
for BrainHybridNet checkpoints.

    python app.py
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr
from hybrid.data import MEAN, STD, eval_transform
from hybrid.model import CLASSES, BrainHybridNet, load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPTS = [p for p in os.environ.get("MODEL_CKPTS", "").split(",") if p] or sorted(glob.glob("checkpoints/*.pt"))
MODELS = [load_checkpoint(p, DEVICE) for p in CKPTS]
IMG_SIZE = MODELS[0][1]["img_size"] if MODELS else 224
TRANSFORM = eval_transform(IMG_SIZE)
DISCLAIMER = ("Research/educational tool only. Not a medical device and not a diagnosis — "
              "results must be reviewed by a qualified radiologist.")


def grad_cam(model, x):
    """Grad-CAM on the stride-16 CNN feature map feeding the Transformer."""
    feats = {}
    handle = model.proj16.register_forward_pre_hook(lambda m, inp: feats.update(f=inp[0]))
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        logits = model(x)
        f = feats["f"]
        grads = torch.autograd.grad(logits[0, logits.argmax()], f)[0]
    handle.remove()
    cam = F.relu((grads.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay(x, cam):
    img = x[0].cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    heat = np.stack([cam, cam * 0.3, 1 - cam], -1)  # simple blue->red map without extra deps
    return (np.clip(0.6 * img + 0.4 * heat, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def predict_one(img):
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.stack([m(x).softmax(-1)[0] for m, _ in MODELS]).mean(0).cpu().numpy()
    hybrid = next((m for m, _ in MODELS if isinstance(m, BrainHybridNet)), None)
    vis = overlay(x, grad_cam(hybrid, x)) if hybrid is not None else np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    return probs, vis


def analyze(files):
    if not MODELS:
        raise gr.Error("No trained checkpoint found in ./checkpoints. Run `python -m hybrid.train` first.")
    if not files:
        raise gr.Error("Upload at least one MRI image.")
    gallery, rows, last_probs = [], [], None
    for f in files:
        path = f if isinstance(f, str) else f.name
        probs, vis = predict_one(Image.open(path))
        pred = CLASSES[int(probs.argmax())]
        gallery.append((vis, f"{pred} ({probs.max() * 100:.1f}%)"))
        rows.append([os.path.basename(path), pred, *[round(float(p) * 100, 2) for p in probs]])
        last_probs = probs
    label = {c: float(p) for c, p in zip(CLASSES, last_probs)}
    return gallery, rows, label


with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown("# Brain Tumor MRI Classifier\nHybrid CNN + Transformer "
                f"({', '.join(c['arch'] for _, c in MODELS) or 'no model loaded'}) "
                f"· classes: {', '.join(CLASSES)}")
    gr.Markdown(f"> ⚠️ {DISCLAIMER}")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="MRI images (one or many)", file_count="multiple",
                            file_types=["image"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Prediction + Grad-CAM", columns=3, height="auto")
            label = gr.Label(label="Probabilities (last image)", num_top_classes=4)
    table = gr.Dataframe(headers=["file", "prediction", *[f"{c} %" for c in CLASSES]],
                         label="All results")
    btn.click(analyze, files, [gallery, table, label])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name=os.environ.get("HOST", "127.0.0.1"))
