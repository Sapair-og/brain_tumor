"""Lightweight CNN + Transformer hybrid models for brain MRI classification."""
import torch
import torch.nn as nn
import timm

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Pretrained hybrid CNN/Transformer backbones from timm that can be ensembled with BrainHybridNet.
TIMM_HYBRIDS = {
    "mobilevitv2": "mobilevitv2_100",          # MobileNet blocks + linear-attention transformer
    "efficientformerv2": "efficientformerv2_s1",  # conv stages + MHSA in late stages
    "fastvit": "fastvit_t8",                   # reparameterised conv + attention-style token mixing
}
ARCHS = ["brainhybrid", *TIMM_HYBRIDS]


class BrainHybridNet(nn.Module):
    """EfficientNet-B0 local features -> Transformer encoder global context -> classifier.

    The CNN extracts multi-scale feature maps (stride 16 and 32). Both are projected to
    tokens, tagged with a scale embedding and positional embedding, and mixed by a small
    Transformer encoder so the model can relate a lesion to the whole brain anatomy.
    """

    def __init__(self, num_classes=len(CLASSES), img_size=224, dim=192, depth=4,
                 heads=4, mlp_ratio=4.0, dropout=0.1, pretrained=True):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                     features_only=True, out_indices=(3, 4))
        c16, c32 = self.cnn.feature_info.channels()
        self.proj16 = nn.Sequential(nn.Conv2d(c16, dim, 1, bias=False), nn.BatchNorm2d(dim))
        self.proj32 = nn.Sequential(nn.Conv2d(c32, dim, 1, bias=False), nn.BatchNorm2d(dim))

        n16, n32 = (img_size // 16) ** 2, (img_size // 32) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n16 + n32, dim))
        self.scale_embed = nn.Parameter(torch.zeros(2, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim * 2, num_classes))

    def forward(self, x):
        f16, f32 = self.cnn(x)
        t16 = self.proj16(f16).flatten(2).transpose(1, 2) + self.scale_embed[0]
        t32 = self.proj32(f32).flatten(2).transpose(1, 2) + self.scale_embed[1]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, t16, t32], dim=1) + self.pos_embed
        tokens = self.norm(self.encoder(tokens))
        # CLS token + mean of patch tokens: stable on small medical datasets.
        feats = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        return self.head(feats)


def build_model(arch="brainhybrid", num_classes=len(CLASSES), pretrained=True, img_size=224):
    if arch == "brainhybrid":
        return BrainHybridNet(num_classes, img_size=img_size, pretrained=pretrained)
    if arch in TIMM_HYBRIDS:
        return timm.create_model(TIMM_HYBRIDS[arch], pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}")


class Ensemble(nn.Module):
    """Averages softmax probabilities of several trained hybrid models."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x).softmax(-1) for m in self.models]).mean(0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False,
                        img_size=ckpt["img_size"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
