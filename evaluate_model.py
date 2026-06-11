"""
Run this from your project root:
    python evaluate_model.py

It will print accuracy, inference time, and resume-ready numbers.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import time
import os
import random

# ── Import your model definition ──────────────────────────────
from disease_detection_v2 import CNN_NeuralNet

# ── Config ────────────────────────────────────────────────────
MODEL_PATH   = 'plant_disease_model.pth'

# Point this to your test/valid folder from the PlantVillage dataset
# Expected structure: DATA_DIR/ClassName/image.jpg
DATA_DIR     = '/Users/kartikjhamb/Downloads/archive (2) 2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'   # ← CHANGE THIS

BATCH_SIZE   = 32
NUM_SAMPLES  = 500   # how many images to test (use None for all)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ── Transform (same as training) ──────────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def main():
    print(f"\n{'='*50}")
    print("  Plant Disease Model Evaluator")
    print(f"{'='*50}\n")
    print(f"Device: {DEVICE}")

    # ── Load model ────────────────────────────────────────────
    print("Loading model...")
    model = CNN_NeuralNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    # ── Count parameters ──────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Inference time (single image) ─────────────────────────
    print("\nMeasuring inference time (single image, CPU)...")
    dummy = torch.randn(1, 3, 256, 256)
    cpu_model = CNN_NeuralNet()
    cpu_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    cpu_model.eval()

    # warmup
    for _ in range(3):
        with torch.no_grad():
            cpu_model(dummy)

    times = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            cpu_model(dummy)
        times.append(time.time() - start)

    avg_ms = sum(times) / len(times) * 1000
    print(f"Average inference time (CPU): {avg_ms:.1f} ms  (~{avg_ms/1000:.2f}s per image)")

    # ── Dataset accuracy ──────────────────────────────────────
    if not os.path.exists(DATA_DIR) or DATA_DIR == 'path/to/your/test_data':
        print("\n⚠️  DATA_DIR not set or doesn't exist — skipping accuracy test.")
        print("   Set DATA_DIR in this script to your PlantVillage test folder.")
    else:
        print(f"\nLoading dataset from: {DATA_DIR}")
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

        if NUM_SAMPLES and NUM_SAMPLES < len(dataset):
            indices = random.sample(range(len(dataset)), NUM_SAMPLES)
            dataset = Subset(dataset, indices)
            print(f"Randomly sampling {NUM_SAMPLES} images for evaluation.")

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        start_all = time.time()

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                print(f"  Processed {total}/{len(dataset)} images...", end='\r')

        elapsed = time.time() - start_all
        accuracy = correct / total * 100

        print(f"\n\nResults on {total} images:")
        print(f"  Accuracy:        {accuracy:.2f}%")
        print(f"  Correct:         {correct}/{total}")
        print(f"  Total time:      {elapsed:.1f}s")

    # ── Resume-ready summary ───────────────────────────────────
    print(f"\n{'='*50}")
    print("  RESUME-READY NUMBERS")
    print(f"{'='*50}")
    print(f"  • Trained on 87,000+ images (PlantVillage dataset)")
    print(f"  • 38 disease classes across 14 plant species")
    print(f"  • Model parameters: {total_params:,}")
    print(f"  • Inference time (CPU): ~{avg_ms:.0f}ms per image")
    if 'accuracy' in dir():
        print(f"  • Test accuracy: {accuracy:.1f}%")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()