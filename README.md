🌳 BonsAI — Bonsai Species & Health Classifier
![Python](https://python.org)
![PyTorch](https://pytorch.org)
![License](LICENSE)
![AMD ROCm](https://rocm.docs.amd.com)
> An AI-powered image classification system for bonsai species identification and plant health assessment, optimized for AMD GPUs using ROCm.
---
📖 Background
Bonsai cultivation is a centuries-old horticultural art form practiced by millions worldwide. However, correctly identifying bonsai species and diagnosing plant health issues remains a significant challenge — especially for beginners and small-scale nursery owners in Southeast Asia, where the bonsai hobby is rapidly growing.
Current challenges faced by bonsai enthusiasts:
Species misidentification leads to improper care (wrong soil, watering, pruning schedules)
Late disease detection causes irreversible damage to trees that took decades to cultivate
Lack of accessible tools — most existing plant AI tools do not specialize in bonsai
BonsAI addresses these problems by providing a lightweight, GPU-accelerated image classifier that can:
Identify bonsai species from a photo
Assess the health condition of a bonsai tree
Suggest care recommendations based on the prediction
---
🎯 Project Goals
Goal	Description
Species Classification	Identify 10+ common bonsai species with ≥85% accuracy
Health Assessment	Detect 4 health conditions (Healthy, Overwatered, Underwatered, Pest Detected)
GPU Acceleration	Train models efficiently using AMD GPU via ROCm
Accessibility	Provide a simple inference script usable without deep ML knowledge
---
🌿 Supported Species
#	Species	Common Name
1	Juniperus chinensis	Chinese Juniper
2	Ficus retusa	Banyan Fig
3	Acer palmatum	Japanese Maple
4	Pinus thunbergii	Japanese Black Pine
5	Rhododendron indicum	Satsuki Azalea
6	Carmona retusa	Fukien Tea
7	Serissa japonica	Snow Rose
8	Ulmus parvifolia	Chinese Elm
9	Portulacaria afra	Jade Bonsai
10	Premna obtusifolia	Pemphis Bonsai
---
🏥 Health Conditions
Condition	Description
✅ Healthy	Normal foliage color, proper leaf density
�💧 Overwatered	Yellowing leaves, root rot indicators
🏜️ Underwatered	Dry/crispy leaf edges, soil pulling away from pot
🐛 Pest Detected	Visible pest damage, discoloration, webbing
---
🏗️ Architecture
```
BonsAI/
├── Backbone:     ResNet-50 (pretrained on ImageNet)
├── Head:         Custom dual-output classifier
│   ├── Species branch   → 10 classes (softmax)
│   └── Health branch    → 4 classes (softmax)
├── Optimizer:    AdamW + Cosine LR Scheduler
├── Loss:         Cross-Entropy (weighted for imbalance)
└── Framework:    PyTorch + ROCm (AMD GPU)
```
The model uses multi-task learning — a single forward pass produces both species and health predictions simultaneously, reducing inference time by ~40% compared to running two separate models.
---
⚙️ Requirements
Hardware
AMD GPU (RX 6000 / RX 7000 series, or AMD Instinct MI-series recommended)
Minimum 8GB VRAM for training (batch size 32)
16GB RAM minimum
Software
```
Python >= 3.10
ROCm >= 5.4
PyTorch >= 2.0 (ROCm build)
```
---
🚀 Installation
1. Clone the Repository
```bash
git clone https://github.com/YOUR\_USERNAME/bonsai-classifier.git
cd bonsai-classifier
```
2. Install ROCm (AMD GPU Support)
```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install\_5.7.50700-1\_all.deb
sudo dpkg -i amdgpu-install\_5.7.50700-1\_all.deb
sudo amdgpu-install --usecase=rocm
```
3. Install Python Dependencies
```bash
pip install -r requirements.txt
```
4. Verify GPU is Detected
```bash
python scripts/check\_gpu.py
```
---
📂 Dataset
We are building a custom dataset of bonsai images collected from:
Public domain photography websites
Open horticultural datasets
Manual photography (controlled conditions)
Current dataset size: ~2,000 images (growing)  
Target dataset size: 10,000+ images
Dataset directory structure:
```
data/
├── train/
│   ├── juniper/
│   ├── ficus/
│   └── ...
├── val/
└── test/
```
To prepare your own dataset, run:
```bash
python scripts/prepare\_dataset.py --input /path/to/raw\_images --output data/
```
---
🏋️ Training
```bash
# Basic training (auto-detects AMD GPU)
python src/train.py --epochs 50 --batch-size 32

# With custom config
python src/train.py \\
  --epochs 100 \\
  --batch-size 64 \\
  --lr 0.001 \\
  --data-dir data/ \\
  --save-dir models/
```
Training logs and checkpoints are saved to `models/`.
---
🔍 Inference
```bash
# Classify a single image
python src/predict.py --image /path/to/bonsai.jpg

# Example output:
# Species:  Ficus retusa (Banyan Fig) — 94.2% confidence
# Health:   Healthy — 87.6% confidence
# Recommendation: Maintain current watering schedule. Prune in early spring.
```
---
📊 Results (Preliminary)
> Note: These are early experimental results on a small dataset. Full benchmarks coming soon.
Metric	Species Task	Health Task
Accuracy	83.4%	79.1%
Precision	81.2%	77.8%
Recall	82.9%	78.3%
Trained on 1,500 images, AMD RX 6700 XT, ~45 minutes training time
---
🗺️ Roadmap
[x] Project structure and baseline model
[x] Data pipeline and preprocessing scripts
[ ] Full dataset collection (10,000 images)
[ ] Model optimization (quantization for edge devices)
[ ] REST API wrapper for web integration
[ ] Mobile app (Android/iOS) for field use
[ ] Multi-language support (Indonesian, Japanese, Chinese)
[ ] Integration with IoT sensors (soil moisture, humidity)
---
🤝 Contributing
Contributions are welcome! If you have bonsai photos you'd like to contribute to the dataset, or want to help improve the model, please:
Fork this repository
Create a feature branch (`git checkout -b feature/your-feature`)
Commit your changes
Open a Pull Request
---
📄 License
This project is licensed under the MIT License — see LICENSE for details.
---
📬 Contact
Project maintained by a bonsai enthusiast learning machine learning.  
Contributions and feedback are very welcome!
---
"Bonsai is the art of patience. AI is the art of learning from data. Together, they grow."
