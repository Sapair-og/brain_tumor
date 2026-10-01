import os

import numpy as np
import pytest
import torch
from PIL import Image

from hybrid.model import ARCHS, CLASSES, BrainHybridNet, Ensemble, build_model, count_params, load_checkpoint


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shape(arch):
    model = build_model(arch, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, len(CLASSES))


def test_brainhybrid_is_lightweight():
    # ResNet50V2 + the original dense head is ~24.6M params.
    assert count_params(BrainHybridNet(pretrained=False)) < 8e6


def test_ensemble_returns_probabilities():
    ens = Ensemble([build_model("brainhybrid", pretrained=False), build_model("fastvit", pretrained=False)]).eval()
    with torch.no_grad():
        p = ens(torch.randn(3, 3, 224, 224))
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("brainhybrid", pretrained=False).eval()
    path = tmp_path / "m.pt"
    torch.save({"arch": "brainhybrid", "classes": CLASSES, "img_size": 224,
                "state_dict": model.state_dict()}, path)
    loaded, _ = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_data_split_removes_leakage(tmp_path):
    from hybrid.data import make_splits
    rng = np.random.default_rng(0)
    for split in ["Training", "Testing"]:
        for c in CLASSES:
            os.makedirs(tmp_path / split / c)
    for c in CLASSES:
        for i in range(10):
            img = Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8))
            img.save(tmp_path / "Training" / c / f"{i}.png")
            if i < 2:  # leaked into test
                img.save(tmp_path / "Testing" / c / f"{i}.png")
        # exact duplicate inside Training
        Image.open(tmp_path / "Training" / c / "5.png").save(tmp_path / "Training" / c / "dup.png")
    train, val, test = make_splits(str(tmp_path))
    test_paths = {os.path.basename(p) for p, _, _ in test}
    assert len(test) == 2 * len(CLASSES)
    assert len(train) + len(val) == 8 * len(CLASSES)
    hashes = [h for _, _, h in train + val]
    assert len(hashes) == len(set(hashes))
    assert not {h for _, _, h in test} & set(hashes)
    assert test_paths == {"0.png", "1.png"}
