# 🌿 AI-Based Plant Leaf Disease Detection & Guidance Agent

An end-to-end deep learning web app that detects plant leaf diseases from photos and gives AI-powered treatment guidance. Built with YOLOv8, Clasification models and Streamlit — with optional Google Gemini integration for detailed advice.

**Live demo :** https://aibasedplantdetection.streamlit.app/

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/Kartik-1818/AI-BASED-PLANT-LEAF-DETECTION.git
cd AI-BASED-PLANT-LEAF-DETECTION

# 2. (Recommended) Virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Gemini API key — one-time setup
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Open secrets.toml and paste: GEMINI_API_KEY = "your-key-here"

# 5. Run
streamlit run Plant_Disease_Agent.py
```

App opens at `http://localhost:8501`. You only set the Gemini key **once** — it lives in `secrets.toml`, not something users re-enter each time. Without a key, the app still works using built-in short advice (Gemini is optional, not required).

---

## ✨ Features

- **Leaf Detection** — YOLOv8 locates and crops the leaf region from any uploaded photo
- **Disease Classification** — Custom CNN classifies 38 disease conditions across 14 plant species
- **AI Guidance** — Built-in short advice always available; Gemini gives detailed, image-grounded treatment plans
- **Multilingual UI** — English and Hindi
- **Results Dashboard** — Every detection logs to CSV, visualized on a separate analytics page
- **Codespaces Ready** — Zero-setup launch via `.devcontainer`

---

## 🤖 How AI Is Used (Build + Runtime)

**At runtime:** Gemini is called with the cropped leaf image plus the CNN's predicted plant/condition, and returns a structured response (diagnosis confirmation → symptoms → action plan → prevention). The prompt is explicitly different for "short" vs "detailed" mode — short mode isn't a truncated version of the long one, it's a separate prompt. If no API key is set, the app falls back to built-in offline guidance instead of failing.

**While building:** AI coding tools were used for specific tasks — debugging the YOLOv8 → crop → CNN pipeline, scaffolding the analytics dashboard page, and migrating the Gemini integration from the deprecated `google-generativeai` SDK to the current `google-genai` SDK. Architecture choices (residual CNN blocks, confidence thresholds, the optional-Gemini-with-fallback design) were made and tuned manually.

---

## 🌱 Supported Plants & Diseases (38 classes, 14 species)

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

Trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Leaf Detection | YOLOv8 (Ultralytics) |
| Disease Classification | Custom CNN (PyTorch, ResNet-style) |
| AI Guidance | Google Gemini API (`google-genai`, optional) |
| Image Processing | OpenCV (headless), Pillow |
| Data / Analytics | Pandas, NumPy |

---

## 🧠 Model Architecture

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

Skip connections in each residual block improve gradient flow; batch normalization follows every convolution.

---

## 📊 How It Works

1. Upload a clear leaf photo
2. YOLOv8 detects and crops the leaf
3. CNN classifies it into one of 38 classes
4. Guidance shown instantly (Gemini optional for detail)
5. Result logged to `results/detection_results.csv` and viewable on the analytics page

---

## ⚙️ App Settings

| Setting | Description |
|---|---|
| Use Gemini Guidance | Toggle detailed AI advice (needs API key) |
| Guidance Mode | Short vs Detailed |
| Language | English / Hindi |
| Leaf Detection Confidence | Min YOLOv8 confidence |
| Min Disease Confidence | Min CNN confidence to report a disease |
| Max Leaves To Check | Limit per image |
| Crop Padding | Padding around detected leaf box |
| Plant Type (optional) | Helps disambiguate similar diseases |

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repo to GitHub — make sure `requirements.txt`, `Plant_Disease_Agent.py`, and the model files (`leaf_detection_model.pt`, `plant_disease_model.pth`) are all committed.
   - ⚠️ If `.gitignore` has `*.pt` / `*.pth`, force-add the model files: `git add -f leaf_detection_model.pt plant_disease_model.pth`
   - ⚠️ Make sure `secrets.toml` is **never** committed (should be in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Fill in:
   - Repository: `https://github.com/Kartik-1818/AI-BASED-PLANT-LEAF-DETECTION`
   - Branch: `main`
   - Main file path: `Plant_Disease_Agent.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   ```
5. Click **Deploy** and watch the build log.

If deploy says *"This repository does not exist"*, it's almost always a GitHub App permission issue, not a URL typo — check GitHub → Settings → Applications → Streamlit → Configure, and confirm this repo is checked under repository access.

---

## Known Limitations

- Trained on **PlantVillage lab-style images** — real-field photos with messy backgrounds may reduce accuracy
- Only **14 species** supported (no wheat, rice, sugarcane yet — dataset limitation, not architectural)
- Very low-res or blurry images may fail leaf detection

---

## Roadmap

- [ ] Field-collected images for Indian crops (wheat, rice, sugarcane)
- [ ] User feedback loop for continuous model improvement
- [ ] Mobile-optimized UI
- [ ] Public Streamlit Cloud deployment

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — Kaggle
- [Ultralytics YOLOv8](https://ultralytics.com/) — Leaf detection backbone
- [Streamlit](https://streamlit.io/) — Web app framework
- [Google Gemini](https://aistudio.google.com/) — AI guidance engine

---

## 📄 License

Open-source. See [LICENSE](LICENSE) for details.
