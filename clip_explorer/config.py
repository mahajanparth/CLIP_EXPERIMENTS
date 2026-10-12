"""Explicit model and reproducibility configuration."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ViT-B/32"
MODEL_REVISION = "a1d071733d7111c9c014f024669f959182114e33"
MODEL_DIR = ROOT / ".cache" / "clip"
CHECKPOINT = MODEL_DIR / "ViT-B-32.pt"
SEED = 2021
TEMPLATES = ("a photo of a {}", "a photograph of a {}", "a drawing of a {}", "a close-up of a {}", "a picture of a {}")
