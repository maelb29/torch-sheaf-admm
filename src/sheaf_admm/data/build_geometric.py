"""Generate a small geometric-shapes classification dataset.

The images use the same cached format as MNIST, so they can be loaded with
``ImageDataset`` and trained with ``MNISTTask`` configured for 4x4 patches.

Usage:
    python -m sheaf_admm.data.build_geometric --output-dir datasets/geometric
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class GeometricConfig:
    train_size: int = 2000
    test_size: int = 500
    noise: float = 0.05
    seed: int = 0
    output_dir: Path = Path("datasets/geometric")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def generate_shapes_dataset(
    n_samples: int = 2000,
    noise: float = 0.05,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four flattened 4x4 patches and their shape labels per image."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    X = torch.zeros((n_samples, 8, 8))
    y = torch.randint(0, 4, (n_samples,), generator=generator)

    for i, label in enumerate(y):
        if label == 0:  # Croix '+'
            X[i, 1:7, 3:5] = 1.0
            X[i, 2:4, 1:7] = 1.0
        elif label == 1:  # Lettre 'T'
            X[i, 1:3, 1:7] = 1.0
            X[i, 1:7, 3:5] = 1.0
        elif label == 2:  # Carré creux
            X[i, 1:7, 1] = X[i, 1:7, 6] = 1.0
            X[i, 1, 1:7] = X[i, 6, 1:7] = 1.0
        elif label == 3:  # Diagonale '\'
            for k in range(1, 7):
                X[i, k, k] = 1.0

    random_noise = torch.randn(X.shape, generator=generator)
    X = torch.clamp(X + random_noise * noise, 0.0, 1.0)
    patches = X.unfold(1, 4, 4).unfold(2, 4, 4).contiguous().view(n_samples, 4, 16)
    return patches, y


def _patches_to_images(patches: torch.Tensor) -> np.ndarray:
    """Reassemble row-major 4x4 patches into NHWC 8x8 single-channel images."""
    images = patches.reshape(-1, 2, 2, 4, 4).permute(0, 1, 3, 2, 4).contiguous()
    images = images.reshape(-1, 8, 8)
    return images.numpy()[..., None].astype(np.float32, copy=False)


def _save_split(root: Path, split: str, images: np.ndarray, labels: np.ndarray) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "images.npy", images)
    np.save(split_dir / "labels.npy", labels)


def build(cfg: GeometricConfig) -> None:
    """Generate and cache train/test splits in the ``ImageDataset`` format."""
    if cfg.train_size <= 0 or cfg.test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes: dict[str, int] = {}
    for split, count, seed in (
        ("train", cfg.train_size, cfg.seed),
        ("test", cfg.test_size, cfg.seed + 1),
    ):
        generator = torch.Generator().manual_seed(seed)
        patches, labels = generate_shapes_dataset(count, cfg.noise, generator=generator)
        images = _patches_to_images(patches)
        labels_array = labels.numpy()
        _save_split(cfg.output_dir, split, images, labels_array)
        split_sizes[split] = count

    metadata = {
        "task": "geometric",
        "image_shape": [8, 8, 1],
        "num_classes": 4,
        "seed": cfg.seed,
        "noise": cfg.noise,
        "splits": split_sizes,
        "split_shapes": {split: [8, 8, 1] for split in split_sizes},
        "config": cfg.to_dict(),
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Geometric dataset cached to {cfg.output_dir}; splits: {list(split_sizes)}")


def parse_args() -> GeometricConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=GeometricConfig.train_size)
    parser.add_argument("--test-size", type=int, default=GeometricConfig.test_size)
    parser.add_argument("--noise", type=float, default=GeometricConfig.noise)
    parser.add_argument("--seed", type=int, default=GeometricConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=GeometricConfig.output_dir)
    args = parser.parse_args()
    return GeometricConfig(
        train_size=args.train_size,
        test_size=args.test_size,
        noise=args.noise,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    build(parse_args())