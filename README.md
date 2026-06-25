# 🌿 AI-Based Plant Leaf Disease Detection & Guidance Agent

An end-to-end deep learning web application that detects plant leaf diseases from photos and provides AI-powered treatment guidance. Built with PyTorch, YOLOv8, and Streamlit — with optional Google Gemini integration for detailed agronomic advice.

---

## ✨ Features

- **Leaf Detection** — YOLOv8 model automatically locates and crops leaf regions from uploaded images
- **Disease Classification** — Custom CNN classifies 38 disease conditions across 14 plant species
- **AI Guidance** — Short built-in advice for every prediction; optional Gemini integration for detailed treatment plans
- **Multilingual UI** — Supports English and Hindi interfaces
- **Results Dashboard** — Logs every detection to CSV; a dedicated analytics page (`Results Analysis`) visualizes history with charts and a color-coded map
- **Codespaces Ready** — `.devcontainer` config launches the app automatically in GitHub Codespaces

---

## 🌱 Supported Plants & Diseases

The model is trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and recognizes **38 classes** across **14 species**:

| Plant | Conditions |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🏗️ Project Structure

```
AI-BASED-PLANT-LEAF-DETECTION-main/
│
├── Plant_Disease_Agent.py      # Main Streamlit app (entry point)
├── disease_detection_v2.py     # CNN model architecture & inference logic
├── evaluate_model.py           # Offline model evaluation script
│
├── pages/
│   └── 2_Results_Analysis.py  # Streamlit analytics dashboard page
│
├── results/
│   └── detection_results.csv  # Auto-generated detection history log
│
├── .streamlit/
│   └── secrets.example.toml   # Template for Gemini API key config
│
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces configuration
│
├── requirements.txt            # Python dependencies
└── runtime.txt                 # Python version pin
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Leaf Detection | YOLOv8 (Ultralytics) |
| Disease Classification | Custom CNN (PyTorch + ResNet-style blocks) |
| AI Guidance | Google Gemini API (optional) |
| Image Processing | OpenCV, Pillow |
| Model Download | gdown (Google Drive) |
| Data / Analytics | Pandas, NumPy |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-BASED-PLANT-LEAF-DETECTION.git
cd AI-BASED-PLANT-LEAF-DETECTION
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API key

For detailed AI-powered treatment guidance, add a free Gemini key from [aistudio.google.com](https://aistudio.google.com/):

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Edit secrets.toml and add your key
```

### 4. Launch the app

```bash
streamlit run Plant_Disease_Agent.py
```

The app will open at `http://localhost:8501`. Pre-trained models are downloaded automatically from Google Drive on first run.

---

## ☁️ Run in GitHub Codespaces (Zero Setup)

Click **Code → Open with Codespaces** on the repository page. The devcontainer will install all dependencies and launch the Streamlit app automatically — no local setup required.

---

## 🧠 Model Architecture

The disease classifier (`CNN_NeuralNet`) is a custom ResNet-style CNN:

```
Input (3 × 256 × 256)
  → Conv Block (64 filters)
  → Conv Block + MaxPool (128 filters)
  → Residual Block (128 filters)
  → Conv Block + MaxPool (256 filters)
  → Conv Block + MaxPool (512 filters)
  → Residual Block (512 filters)
  → GlobalMaxPool → Flatten → Linear(512 → 38)
```

Each residual block uses skip connections to improve gradient flow and prevent degradation during training. Batch normalization is applied after every convolution.

---

## 📊 How It Works

1. **Upload** a clear photo of a plant leaf
2. **YOLOv8** detects and crops the leaf region(s) from the image
3. **CNN** classifies each cropped leaf into one of 38 disease/healthy classes
4. **Guidance** is displayed immediately; enable Gemini for detailed treatment steps
5. **Results** are logged to `results/detection_results.csv` and visible in the Results Analysis page

---

## ⚙️ App Settings

| Setting | Description |
|---|---|
| Use Gemini Guidance | Toggle detailed AI advice (requires API key) |
| Guidance Mode | Short summary vs. detailed treatment plan |
| Language | English / Hindi |
| Leaf Detection Confidence | Minimum YOLOv8 confidence threshold |
| Min Disease Confidence | Minimum CNN confidence to report a disease |
| Max Leaves To Check | Limit detections per image |
| Crop Padding | Padding around detected leaf bounding box |
| Plant Type (optional) | Provide context to improve accuracy |

---

## ⚠️ Known Limitations

- The CNN is trained on **PlantVillage benchmark images** (controlled, lab-style photos). Real-field images with complex backgrounds may reduce accuracy.
- Only **14 plant species** are supported. Wheat, rice, sugarcane, and other staple crops are not yet included — this is a dataset limitation, not an architectural one.
- Very low-resolution or heavily blurred images may fail leaf detection.

---

## 🗺️ Roadmap

- [ ] Field-collected images for Indian crops (wheat, rice, sugarcane)
- [ ] User feedback loop for continuous model improvement
- [ ] Mobile-optimized UI for low-end Android devices
- [ ] Public Streamlit Cloud deployment

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — Kaggle
- [Ultralytics YOLOv8](https://ultralytics.com/) — Leaf detection backbone
- [Streamlit](https://streamlit.io/) — Web app framework
- [Google Gemini](https://aistudio.google.com/) — AI guidance engine

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.
