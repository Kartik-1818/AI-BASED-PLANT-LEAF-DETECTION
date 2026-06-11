# AI Plant Disease Detection & Intelligent Agent 🌱🔬

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO-blueviolet.svg)](https://ultralytics.com/)
[![GenAI](https://img.shields.io/badge/GenAI-Gemini-blue.svg)](https://aistudio.google.com/)

An **AI-powered Plant Disease Detection & Intelligent Agent** that analyzes plant leaf images to identify diseases across 38 classes and delivers expert agricultural guidance — powered by a two-stage CV pipeline and Google Gemini AI.

## 🚀 Key Features

- **Local Image Upload:** Upload leaf images directly from your device for instant analysis.
- **Two-Stage Detection Pipeline:**
  1. **Leaf Detection:** YOLO model localizes and crops the leaf region from the image.
  2. **Disease Classification:** Custom CNN classifies the cropped leaf into one of **38 plant/disease categories** across 10+ species (Apple, Tomato, Potato, Grape, Corn, and more).
- **🤖 AI Agent Guidance (Gemini):** Optionally integrates Google Gemini to:
  - Visually verify and correct the CNN's prediction using the actual image.
  - Explain detected conditions in plain language.
  - Provide immediate crop-saving action steps.
  - Suggest long-term prevention measures.
  - Assess risk if the condition goes untreated.
- **Offline Guidance Mode:** Works fully without a Gemini API key — built-in expert guidance for all 38 disease classes, no internet required.
- **Multilingual Support:** Full UI and guidance available in **English and Hindi (हिंदी)**.
- **Results History & Analysis:** Every detection is logged with timestamp, plant type, condition, and confidence. View statistics, trends, and condition breakdowns on the Results Analysis page.

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| Core | Python 3.8+, PyTorch, Streamlit |
| Computer Vision | Ultralytics YOLO, OpenCV, PIL |
| Disease Classification | Custom CNN (38 classes, trained on PlantVillage dataset) |
| Generative AI (optional) | Google Gemini (auto-selects best available model) |
| Data & Analysis | Pandas, NumPy |

## ⚙️ Setup & Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/your-username/plant-disease-agent.git
   cd plant-disease-agent
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **(Optional) Enable Gemini AI guidance:**
   - Copy `.streamlit/secrets.example.toml` → `.streamlit/secrets.toml`
   - Add your Gemini API key:
```toml
     GEMINI_API_KEY = "your_key_here"
```
   - Free keys available at [aistudio.google.com](https://aistudio.google.com/)

4. **Run the app:**
```bash
   streamlit run Plant_Disease_Agent.py
```

## 📖 Usage
1. **AI Agent page:** Upload a clear photo of a plant leaf → click **Analyze Leaf**.
   - Offline mode: instant guidance from built-in expert rules.
   - Gemini mode: toggle **"Use Gemini Guidance"** for detailed, image-aware analysis.
   - Switch between **English / Hindi** from the sidebar.
2. **Results Analysis page:** Browse detection history, view plant/condition breakdowns, and track detection trends over time.


## 🧠 Models
| Model | Type | Purpose |
|---|---|---|
| `leaf_detection_model.pt` | YOLO (Ultralytics) | Detects and localizes leaf regions in the image |
| `plant_disease_model.pth` | Custom CNN (PyTorch) | Classifies leaf into 1 of 38 disease/healthy categories |

**Supported classes include:** Apple Scab, Black Rot, Cedar Apple Rust, Powdery Mildew, Late Blight, Early Blight, Leaf Mold, Mosaic Virus, Yellow Leaf Curl Virus, Spider Mites, and more — across Apple, Tomato, Potato, Grape, Corn, Peach, Strawberry, Cherry, Orange, and Pepper.

## 📊 Detection Logging

All detections are saved to `results/detection_results.csv` with:
- `timestamp` — when the detection occurred
- `plant_type` — detected plant species
- `condition` — disease name or "healthy"
- `confidence` — model confidence score (0.0 – 1.0)

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — 87,000+ leaf images across 38 classes
- [Ultralytics](https://ultralytics.com/) — YOLO object detection framework
- [Streamlit](https://streamlit.io/) — web app framework
- [Google Gemini](https://aistudio.google.com/) — generative AI guidance


 ## ⚠️ Current Limitations & Roadmap

**Supported plants & diseases:** The current model is trained on the 
PlantVillage dataset (38 classes across 14 plant species). Detection 
outside these classes may produce inaccurate results.

**Planned improvements:**
- [ ] Expand training data with custom field-collected images
- [ ] Add support for region-specific crops (wheat, rice, sugarcane)
- [ ] Integrate user feedback loop to improve model over time
---
*Built for intelligent crop protection and real-time agricultural guidance.*