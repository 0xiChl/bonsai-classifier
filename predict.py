"""
BonsAI - Inference Script
=========================
Load a trained model and classify a bonsai image.

Usage:
    python src/predict.py --image /path/to/bonsai.jpg
    python src/predict.py --image /path/to/bonsai.jpg --model models/bonsai_best.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from train import BonsAIClassifier


# ── Class Labels ──────────────────────────────────────────────────────────────

SPECIES_CLASSES = {
    0: ("Juniperus chinensis",    "Chinese Juniper",   "Water moderately. Full sun. Prune in early spring."),
    1: ("Ficus retusa",           "Banyan Fig",        "Keep indoors. Water when topsoil is dry. Mist leaves."),
    2: ("Acer palmatum",          "Japanese Maple",    "Partial shade. Protect from strong winds. Water regularly."),
    3: ("Pinus thunbergii",       "Japanese Black Pine","Full sun. Water sparingly. Repot every 3-4 years."),
    4: ("Rhododendron indicum",   "Satsuki Azalea",    "Acidic soil. Water regularly. Fertilize after flowering."),
    5: ("Carmona retusa",         "Fukien Tea",        "Warm climate. Moderate watering. Avoid cold drafts."),
    6: ("Serissa japonica",       "Snow Rose",         "Semi-shade. Consistent moisture. Sensitive to change."),
    7: ("Ulmus parvifolia",       "Chinese Elm",       "Adaptable. Regular watering. Hardy and beginner-friendly."),
    8: ("Portulacaria afra",      "Jade Bonsai",       "Full sun. Water sparingly. Tolerates drought well."),
    9: ("Premna obtusifolia",     "Pemphis Bonsai",    "Full sun. Minimal watering. Excellent for tropical climates."),
}

HEALTH_CLASSES = {
    0: ("Healthy",      "✅", "Your bonsai looks great! Maintain your current care routine."),
    1: ("Overwatered",  "💧", "Reduce watering frequency. Ensure proper drainage. Check roots for rot."),
    2: ("Underwatered", "🏜️", "Increase watering frequency. Soak soil thoroughly. Check humidity levels."),
    3: ("Pest Detected","🐛", "Inspect leaves closely. Apply neem oil or insecticidal soap. Isolate the tree."),
}


# ── Image Preprocessing ───────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image_path: str, model_path: str = "models/bonsai_best.pt"):
    # ── Load Image ────────────────────────────────────────────────────────────
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # ── Load Model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BonsAIClassifier(num_species=len(SPECIES_CLASSES), num_health=len(HEALTH_CLASSES))

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  No trained model found at '{model_path}'.")
        print("   Running with random weights (for demo purposes only).")
        print("   Train the model first: python src/train.py")

    model.to(device)
    model.eval()
    tensor = tensor.to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        sp_logits, hl_logits = model(tensor)

    sp_probs = F.softmax(sp_logits, dim=1)[0]
    hl_probs = F.softmax(hl_logits, dim=1)[0]

    sp_idx = sp_probs.argmax().item()
    hl_idx = hl_probs.argmax().item()

    sp_confidence = sp_probs[sp_idx].item() * 100
    hl_confidence = hl_probs[hl_idx].item() * 100

    # ── Output ────────────────────────────────────────────────────────────────
    sp_sci, sp_common, sp_tip = SPECIES_CLASSES[sp_idx]
    hl_name, hl_icon, hl_tip = HEALTH_CLASSES[hl_idx]

    print("\n" + "═" * 55)
    print("  🌳 BonsAI Classification Result")
    print("═" * 55)
    print(f"  Image:   {img_path.name}")
    print("─" * 55)
    print(f"  Species: {sp_sci}")
    print(f"           ({sp_common})")
    print(f"           Confidence: {sp_confidence:.1f}%")
    print("─" * 55)
    print(f"  Health:  {hl_icon} {hl_name}")
    print(f"           Confidence: {hl_confidence:.1f}%")
    print("─" * 55)
    print(f"  Species Tip:  {sp_tip}")
    print(f"  Health Tip:   {hl_tip}")
    print("═" * 55 + "\n")

    # Return as dict for programmatic use
    return {
        "species": {
            "scientific": sp_sci,
            "common":     sp_common,
            "confidence": round(sp_confidence, 2),
            "care_tip":   sp_tip,
        },
        "health": {
            "condition":  hl_name,
            "confidence": round(hl_confidence, 2),
            "advice":     hl_tip,
        }
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify a bonsai image")
    parser.add_argument("--image", required=True, help="Path to bonsai image (JPG/PNG)")
    parser.add_argument("--model", default="models/bonsai_best.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    predict(args.image, args.model)


if __name__ == "__main__":
    main()
