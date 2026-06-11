import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
genai = None
import os
from datetime import datetime
from disease_detection_v2 import CNN_NeuralNet
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

import csv
import gdown
import os

def download_models():
    if not os.path.exists('plant_disease_model.pth'):
        print("Downloading disease model...")
        gdown.download(id="1RbqZwmy-7nL_fs7qp9xvknWucFJ4gaFC", output='plant_disease_model.pth', quiet=False)
    if not os.path.exists('leaf_detection_model.pt'):
        print("Downloading leaf detection model...")
        gdown.download(id="1kZcF8-hJN55-LfDigskhb4ihKy8n_2q9", output='leaf_detection_model.pt', quiet=False)

download_models()
# --- Constants ---
CSV_FILENAME = 'results/detection_results.csv'
LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
}
UI_TEXT = {
    "en": {
        "app_title": "🌿 AI-Based Plant Disease Detection & Guidance Agent",
        "app_subtitle": "Intelligent Crop Protection and Real-time Advice",
        "settings": "Settings",
        "use_gemini": "Use Gemini Guidance",
        "guidance_mode": "Guidance Mode",
        "mode_short": "Short",
        "mode_detailed": "Detailed",
        "language": "Language",
        "leaf_conf": "Leaf Detection Confidence",
        "min_disease_conf": "Min Disease Confidence",
        "max_leaves": "Max Leaves To Check",
        "crop_padding": "Leaf Crop Padding",
        "plant_type_optional": "Plant Type (optional)",
        "how_it_works": "**How it works:**\n1. Upload a clear photo of a plant leaf.\n2. The AI detects the leaf and classifies any potential disease.\n3. Guidance is short by default; enable Gemini for detailed analysis.",
        "upload": "Upload an image of a plant leaf...",
        "analyze": "Analyze Leaf",
        "uploaded": "Uploaded Image",
        "classification": "Classification Result",
        "ai_guidance": "AI Agent Guidance",
        "detected_leaf": "Detected Leaf",
        "top_predictions": "Top predictions:",
        "label_plant": "Plant",
        "label_condition": "Condition",
        "label_confidence": "Confidence",
        "classify_fail": "Could not classify this image. Please try another photo.",
        "low_conf": "Low confidence. Try selecting the correct plant type and re-run with a clearer image.",
        "gemini_not_configured": "Gemini is enabled, but no server API key is configured.",
        "advanced": "Advanced",
        "model": "Gemini Model",
        "load_models": "Load Gemini Models",
    },
    "hi": {
        "app_title": "🌿 एआई आधारित पौधा रोग पहचान एवं मार्गदर्शन",
        "app_subtitle": "फसल सुरक्षा के लिए त्वरित सलाह",
        "settings": "सेटिंग्स",
        "use_gemini": "Gemini मार्गदर्शन चालू करें",
        "guidance_mode": "मार्गदर्शन मोड",
        "mode_short": "संक्षिप्त",
        "mode_detailed": "विस्तृत",
        "language": "भाषा",
        "leaf_conf": "पत्ती पहचान कॉन्फिडेंस",
        "min_disease_conf": "न्यूनतम रोग कॉन्फिडेंस",
        "max_leaves": "अधिकतम पत्तियाँ जाँचें",
        "crop_padding": "लीफ क्रॉप पैडिंग",
        "plant_type_optional": "पौधे का प्रकार (वैकल्पिक)",
        "how_it_works": "**कैसे काम करता है:**\n1. पत्ती की साफ़ फोटो अपलोड करें।\n2. AI पत्ती ढूंढकर रोग वर्गीकृत करता है।\n3. डिफ़ॉल्ट सलाह संक्षिप्त है; विस्तृत विश्लेषण के लिए Gemini चालू करें।",
        "upload": "पौधे की पत्ती की फोटो अपलोड करें...",
        "analyze": "विश्लेषण करें",
        "uploaded": "अपलोड की गई छवि",
        "classification": "वर्गीकरण परिणाम",
        "ai_guidance": "एआई मार्गदर्शन",
        "detected_leaf": "पहचानी गई पत्ती",
        "top_predictions": "शीर्ष अनुमान:",
        "label_plant": "पौधा",
        "label_condition": "स्थिति",
        "label_confidence": "कॉन्फिडेंस",
        "classify_fail": "यह छवि वर्गीकृत नहीं हो सकी। कृपया दूसरी/साफ़ फोटो आज़माएं।",
        "low_conf": "कॉन्फिडेंस कम है। सही पौधा चुनकर और साफ़ फोटो से फिर प्रयास करें।",
        "gemini_not_configured": "Gemini चालू है, लेकिन सर्वर API key सेट नहीं है।",
        "advanced": "एडवांस्ड",
        "model": "Gemini मॉडल",
        "load_models": "Gemini मॉडल लोड करें",
    },
}

def tr(lang, key):
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))

PLANT_HI = {
    "Apple": "सेब",
    "Blueberry": "ब्लूबेरी",
    "Cherry (including sour)": "चेरी",
    "Corn (maize)": "मक्का",
    "Grape": "अंगूर",
    "Orange": "संतरा",
    "Peach": "आड़ू",
    "Pepper, bell": "शिमला मिर्च",
    "Potato": "आलू",
    "Raspberry": "रास्पबेरी",
    "Soybean": "सोयाबीन",
    "Squash": "कद्दू वर्ग (स्क्वैश)",
    "Strawberry": "स्ट्रॉबेरी",
    "Tomato": "टमाटर",
}

DISEASE_HI = {
    "healthy": "स्वस्थ",
    "apple scab": "सेब की स्कैब (पपड़ी रोग)",
    "black rot": "ब्लैक रॉट (काला सड़न)",
    "cedar apple rust": "सीडर-एप्पल रस्ट (जंग रोग)",
    "powdery mildew": "पाउडरी मिल्ड्यू (चूर्णी फफूंदी)",
    "cercospora leaf spot gray leaf spot": "सेरकोस्पोरा/ग्रे लीफ स्पॉट",
    "common rust": "कॉमन रस्ट (जंग)",
    "northern leaf blight": "नॉर्दर्न लीफ ब्लाइट",
    "esca (black measles)": "एस्का (ब्लैक मीज़ल्स)",
    "leaf blight (isariopsis leaf spot)": "लीफ ब्लाइट (इसारिओप्सिस स्पॉट)",
    "haunglongbing (citrus greening)": "हुआंगलोंगबिंग (साइट्रस ग्रीनिंग)",
    "bacterial spot": "बैक्टीरियल स्पॉट (जीवाणु धब्बा)",
    "early blight": "अर्ली ब्लाइट",
    "late blight": "लेट ब्लाइट",
    "leaf scorch": "लीफ स्कॉर्च (पत्ती झुलसन)",
    "leaf mold": "लीफ मोल्ड (पत्ती फफूंदी)",
    "septoria leaf spot": "सेप्टोरिया लीफ स्पॉट",
    "spider mites two-spotted spider mite": "स्पाइडर माइट्स (दो-धब्बे वाला माइट)",
    "target spot": "टार्गेट स्पॉट",
    "tomato yellow leaf curl virus": "टमाटर येलो लीफ कर्ल वायरस",
    "tomato mosaic virus": "टमाटर मोज़ेक वायरस",
}

def normalize_text(s):
    return " ".join((s or "").replace("_", " ").strip().lower().split())

def translate_plant(plant, lang):
    if lang != "hi":
        return plant
    return PLANT_HI.get(plant, plant)

def translate_condition(condition, lang):
    if lang != "hi":
        return condition
    key = normalize_text(condition)
    return DISEASE_HI.get(key, condition)

def translate_pair(plant, condition, lang):
    return translate_plant(plant, lang), translate_condition(condition, lang)

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

# --- Model Loading ---
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        leaf_model_path = 'leaf_detection_model.pt'
        disease_model_path = 'plant_disease_model.pth'
        
        if not os.path.exists(leaf_model_path) or not os.path.exists(disease_model_path):
            st.error("Model files not found. Please ensure 'leaf_detection_model.pt' and 'plant_disease_model.pth' are in the project root.")
            return None, None, None
            
        leaf_model = YOLO(leaf_model_path)
        disease_model = CNN_NeuralNet()
        disease_model.load_state_dict(
            torch.load(disease_model_path, map_location=torch.device('cpu'), weights_only=False)
        )
        disease_model.to(device)
        disease_model.eval()
        
        return leaf_model, disease_model, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- Disease Classification ---
def classify_disease(cropped_leaf, disease_model, device):
    if disease_model is None: return None
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    try:
        if not isinstance(cropped_leaf, np.ndarray) or cropped_leaf.size == 0: return None
        pil_image = Image.fromarray(cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = disease_model(tensor)
            probs = F.softmax(outputs, dim=1)
        return probs.squeeze(0).detach().cpu().numpy()
    except Exception:
        return None

def get_topk_predictions(probs, plant_filter=None, k=3):
    if probs is None or not isinstance(probs, np.ndarray) or probs.size != len(CLASS_NAMES):
        return []
    indices = list(range(len(CLASS_NAMES)))
    if plant_filter and plant_filter != "Auto":
        indices = [i for i in indices if CLASS_NAMES[i].startswith(f"{plant_filter}___")]
    if not indices:
        return []
    sub_probs = probs[indices]
    order = np.argsort(sub_probs)[::-1][:k]
    denom = float(np.sum(sub_probs)) if plant_filter and plant_filter != "Auto" else 1.0
    denom = denom if denom > 0 else 1.0
    out = []
    for j in order:
        i = indices[int(j)]
        raw = float(probs[i])
        score = float(sub_probs[int(j)] / denom)
        out.append((CLASS_NAMES[i], raw, score))
    return out

def get_offline_guidance(plant_type, condition):
    p = (plant_type or "").strip()
    c = " ".join((condition or "").strip().lower().split())
    key = (p.lower(), c)

    disease_specific = {
        ("apple", "apple scab"): """### What it is
- A fungal disease causing olive/black spots on leaves and fruit.

### Immediate actions
- Remove fallen leaves and infected debris; do not compost.
- Prune to improve airflow and keep foliage dry.
- Apply a labeled fungicide (preventive sprays work best) during wet weather periods.

### Prevention
- Use resistant varieties if possible and keep orchard floor clean.""",
        ("apple", "black rot"): """### What it is
- A fungal disease that can affect leaves, fruit, and branches.

### Immediate actions
- Remove mummified fruit and infected twigs/branches.
- Sanitize pruning tools between cuts.
- Apply fungicide during infection periods (warm + humid).

### Prevention
- Reduce tree stress (proper watering/fertilization) and maintain good canopy airflow.""",
        ("apple", "cedar apple rust"): """### What it is
- A rust disease that needs both apple and cedar/juniper hosts.

### Immediate actions
- If possible, remove nearby juniper galls (orange, gelatinous in wet weather).
- Apply labeled rust fungicide early in the season.

### Prevention
- Plant resistant varieties and avoid planting apples near junipers.""",
        ("cherry (including sour)", "powdery mildew"): """### What it is
- A fungal disease that looks like white/gray powder on leaves.

### Immediate actions
- Remove heavily infected leaves/shoots.
- Improve airflow and avoid excess nitrogen.
- Apply a labeled fungicide (sulfur/bicarbonate products are common options).

### Prevention
- Keep canopy open and avoid overhead irrigation.""",
        ("squash", "powdery mildew"): """### Immediate actions
- Remove severely infected leaves but don’t over-defoliate the plant.
- Increase spacing/airflow and water at the base.
- Apply labeled fungicide; start early and repeat as directed.

### Prevention
- Grow resistant varieties and rotate cucurbits yearly.""",
        ("potato", "early blight"): """### Immediate actions
- Remove infected lower leaves; reduce splash by mulching.
- Avoid overhead watering; water early.
- Apply labeled fungicide and repeat during rainy periods.

### Prevention
- Rotate away from potatoes/tomatoes for 2–3 years and destroy volunteer plants.""",
        ("potato", "late blight"): """### Urgent note
- Late blight spreads fast and can destroy a crop quickly.

### Immediate actions
- Remove and destroy infected plants/leaves immediately.
- Apply late-blight labeled fungicide urgently.
- Avoid moving through wet fields to reduce spread.

### Prevention
- Use certified seed, resistant varieties, and strict sanitation.""",
        ("tomato", "early blight"): """### Immediate actions
- Remove lower infected leaves and stake/prune for airflow.
- Mulch to prevent soil splash; water at soil level.
- Apply labeled fungicide during humid/wet periods.

### Prevention
- Rotate crops and remove plant debris at season end.""",
        ("tomato", "late blight"): """### Urgent note
- Late blight can wipe out tomatoes rapidly in cool, wet weather.

### Immediate actions
- Remove infected plants/leaves and isolate the area.
- Apply late-blight labeled fungicide immediately.
- Do not compost infected material.

### Prevention
- Use resistant varieties and preventive sprays when weather risk is high.""",
        ("tomato", "leaf mold"): """### Immediate actions
- Reduce humidity (ventilate greenhouse, increase spacing).
- Remove infected leaves and avoid wetting foliage.
- Apply labeled fungicide if pressure is high.

### Prevention
- Keep leaves dry and improve airflow; sanitize greenhouse surfaces.""",
        ("tomato", "septoria leaf spot"): """### Immediate actions
- Remove infected leaves and keep foliage dry.
- Mulch to reduce soil splash; stake plants.
- Apply labeled fungicide and repeat as directed.

### Prevention
- Rotate crops and clean up debris at end of season.""",
        ("tomato", "spider mites two-spotted spider mite"): """### Immediate actions
- Check leaf undersides; isolate affected plants.
- Spray undersides with water, then use insecticidal soap/neem/miticide as labeled.
- Reduce dust and plant stress (consistent watering).

### Prevention
- Monitor weekly; keep weeds down and avoid drought stress.""",
        ("tomato", "target spot"): """### Immediate actions
- Remove infected leaves and increase airflow.
- Avoid overhead irrigation; water early.
- Apply labeled fungicide if lesions are spreading.

### Prevention
- Sanitation and crop rotation; avoid prolonged leaf wetness.""",
        ("tomato", "tomato yellow leaf curl virus"): """### Important note
- Viral disease; infected plants usually won’t recover.

### Immediate actions
- Control whiteflies aggressively (traps + labeled insecticide).
- Remove severely infected plants to protect the rest.

### Prevention
- Use resistant varieties and insect netting where possible.""",
        ("tomato", "tomato mosaic virus"): """### Important note
- Viral disease; spreads via hands/tools and contaminated material.

### Immediate actions
- Remove infected plants if symptoms are strong.
- Disinfect tools and wash hands; avoid smoking near plants.

### Prevention
- Use certified seed and resistant varieties; strict hygiene.""",
        ("strawberry", "leaf scorch"): """### Immediate actions
- Remove heavily affected leaves and keep rows clean.
- Reduce leaf wetness (drip irrigation, morning watering).
- Apply labeled fungicide if symptoms are increasing.

### Prevention
- Renovate beds after harvest, improve airflow, and avoid overcrowding.""",
        ("orange", "haunglongbing (citrus greening)"): """### Critical note
- Citrus greening is severe and usually not curable once established.

### Immediate actions
- Contact local agricultural extension for confirmation and guidance.
- Control psyllids aggressively; remove infected trees if advised.

### Prevention
- Plant clean nursery stock and maintain vector control."""
    }

    if key in disease_specific:
        return disease_specific[key]

    if c.endswith("healthy") or c == "healthy":
        return f"""### Recommended actions
- Keep monitoring {p} weekly and remove old/damaged leaves.
- Water at soil level (avoid wet leaves) and maintain good airflow.
- Use balanced fertilizer and avoid overwatering.

### Prevention
- Sanitize tools, rotate crops where applicable, and control weeds."""

    if "bacterial spot" in c:
        return """### Immediate actions
- Remove infected leaves and destroy them (do not compost).
- Avoid overhead irrigation; water early and at soil level.
- Sanitize tools; consider copper-based bactericide per label guidance.

### Prevention
- Improve spacing/airflow and rotate crops; use disease-free seedlings."""

    if "virus" in c:
        return """### Important note
- Viral diseases usually have no direct cure once infected.

### Immediate actions
- Remove severely infected plants if spread risk is high.
- Control insect vectors (whiteflies/aphids) and disinfect tools.

### Prevention
- Use resistant varieties and insect-proofing where possible."""

    if "mite" in c:
        return """### Immediate actions
- Inspect leaf undersides; wash leaves with water.
- Use insecticidal soap/neem/miticide per label and repeat as needed.

### Prevention
- Monitor weekly and reduce plant stress (water consistency)."""

    if any(x in c for x in ["blight", "mold", "leaf spot", "scab", "rust", "black rot", "powdery", "leaf scorch"]):
        return f"""### Immediate actions
- Remove infected leaves/fruit; destroy plant debris.
- Improve airflow and avoid wetting foliage.
- Use a labeled fungicide for {p} and repeat as directed.

### Prevention
- Crop rotation, sanitation, and preventive sprays during high-risk weather."""

    return f"""### Immediate actions
- Take 2–3 clearer photos in good daylight (front + back of leaf).
- Remove the most affected leaves and isolate suspicious plants.
- Avoid overhead watering and improve airflow.

### Next step
- Confirm with local extension/lab if symptoms persist on {p}."""

def get_offline_guidance_short(plant_type, condition, lang):
    p = (plant_type or "").strip()
    c = " ".join((condition or "").strip().lower().split())
    if not p:
        p = "this plant" if lang != "hi" else "इस पौधे"
    if "uncertain:" in c:
        c = c.replace("uncertain:", "").strip()

    if c.endswith("healthy") or c == "healthy":
        if lang == "hi":
            return f"""- {p} स्वस्थ लग रहा है।\n- पत्तों को सूखा रखें (जड़ में पानी दें) और हवा का प्रवाह बनाए रखें।\n- हफ्ते में 1 बार जांचें; खराब पत्ते हटाएं।"""
        return f"""- Looks healthy for {p}.\n- Keep leaves dry (water at soil level) and ensure airflow.\n- Recheck weekly; remove damaged leaves early."""

    if any(x in c for x in ["late blight", "haunglongbing", "citrus greening"]):
        if lang == "hi":
            return f"""- {p} के लिए यह हाई-रिस्क स्थिति है: तुरंत कार्रवाई करें।\n- संक्रमित हिस्से अलग करें/हटाएं; गीले पौधों में काम करने से बचें।\n- स्थानीय कृषि विशेषज्ञ/एक्सटेंशन से तुरंत पुष्टि करें और लेबल अनुसार नियंत्रण करें।"""
        return f"""- High-risk condition for {p}: act immediately.\n- Isolate/remove infected parts; avoid moving through wet plants.\n- Confirm with local expert/extension ASAP and follow labeled control measures."""

    if "virus" in c or "mosaic" in c or "yellow leaf curl" in c:
        if lang == "hi":
            return f"""- {p} में वायरल समस्या की संभावना (अक्सर इलाज नहीं होता)।\n- ज्यादा संक्रमित पौधे हटाएं और औज़ार/हाथ सैनिटाइज़ करें।\n- व्हाइटफ्लाई/एफिड जैसे वाहक कीट नियंत्रित करें ताकि फैलाव रुके।"""
        return f"""- Likely viral issue for {p} (often not curable).\n- Remove heavily infected plants and disinfect tools.\n- Control vectors (whiteflies/aphids) to stop spread."""

    if "bacterial" in c:
        if lang == "hi":
            return f"""- {p} में बैक्टीरियल समस्या की संभावना।\n- संक्रमित पत्ते हटाएं; ऊपर से पानी देने से बचें।\n- औज़ार सैनिटाइज़ करें और लेबल अनुसार कॉपर स्प्रे पर विचार करें।"""
        return f"""- Likely bacterial issue for {p}.\n- Remove infected leaves; avoid overhead watering.\n- Sanitize tools and consider copper-based spray per label."""

    if "mite" in c:
        if lang == "hi":
            return f"""- {p} में माइट/कीट का असर संभव।\n- पत्तों के नीचे देखें; पानी से धोएं और लेबल अनुसार साबुन/नीम/माइटिसाइड का उपयोग करें।\n- पौधे का तनाव कम करें (नियमित सिंचाई)।"""
        return f"""- Likely mite/pest stress for {p}.\n- Inspect leaf undersides; wash leaves and use soap/neem/miticide per label.\n- Reduce plant stress (consistent watering)."""

    if lang == "hi":
        return f"""- {p} में {condition} की संभावना।\n- संक्रमित पत्ते/फल हटाएं और पत्तों को सूखा रखें।\n- लक्षण बढ़ें तो फसल-उपयुक्त फफूंदनाशी/आईपीएम उपाय लेबल अनुसार अपनाएं।"""
    return f"""- Suspected {condition} on {p}.\n- Remove infected leaves/fruit and keep foliage dry.\n- Use crop-appropriate fungicide/integrated control per label if symptoms spread."""

def get_gemini_api_key():
    try:
        temp = (st.session_state.get("temp_gemini_api_key", "") or "").strip()
    except Exception:
        temp = ""
    if temp and "PASTE_YOUR_GEMINI_API_KEY_HERE" not in temp:
        return temp
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    key = (key or "").strip()
    if key and "PASTE_YOUR_GEMINI_API_KEY_HERE" not in key:
        return key
    env_key = (os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if env_key and "PASTE_YOUR_GEMINI_API_KEY_HERE" not in env_key:
        return env_key
    return ""

def get_preferred_gemini_model(models):
    preferred = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-latest",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    for p in preferred:
        if p in models:
            return p
    for m in models:
        if "flash" in m:
            return m
    return models[0] if models else ""

# --- CSV Saving ---
def save_to_csv(result, filename=CSV_FILENAME):
    fieldnames = ['timestamp', 'plant_type', 'condition', 'confidence']
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        is_new_file = not os.path.exists(filename) or os.path.getsize(filename) == 0
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            if is_new_file: writer.writeheader()
            writer.writerow(result)
    except Exception as e:
        st.error(f"Error writing to CSV: {e}")

# --- GenAI Recommendations ---
def get_genai_recommendations(cropped_leaf_rgb, plant_type_guess, condition_guess, model_name, mode, lang):
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini is disabled (no server API key configured)."
    
    try:
        global genai
        if genai is None:
            import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Convert the numpy RGB image to a PIL Image for Gemini
        pil_img = Image.fromarray(cropped_leaf_rgb)
        
        if mode == "short":
            prompt = f"""
            You are an expert agricultural consultant and plant pathologist.
            Analyze the provided leaf image carefully and respond in language code: {lang}.
            
            The on-device CNN guessed:
            - Plant: {plant_type_guess}
            - Condition: {condition_guess}
            
            Output must be SHORT (max 6 bullets total):
            - Verified diagnosis (plant + condition) + confidence (high/medium/low)
            - 3 immediate actions (bullets)
            - 1 prevention tip (bullet)
            - 1 escalate/when-to-call-expert tip (bullet)
            """
        else:
            prompt = f"""
            You are an expert agricultural consultant and plant pathologist.
            Analyze the provided leaf image carefully and respond in language code: {lang}.
            
            The on-device CNN guessed:
            - Plant: {plant_type_guess}
            - Condition: {condition_guess}
            
            Your tasks:
            1. Visual verification: confirm/correct plant + condition from the image.
            2. Visual findings: list 3–6 visible symptoms you used to decide.
            3. Action plan (detailed): immediate actions (today/this week) and prevention (next season).
            4. Risk: what happens if untreated + when to escalate (expert/lab).
            
            Output format:
            - Start with: **Verified Diagnosis** (plant + condition)
            - Then: **Confidence** (high/medium/low) + one sentence why
            - Then: **Visible Findings** (bullets)
            - Then: **Immediate Actions** (bullets)
            - Then: **Prevention** (bullets)
            - Then: **Escalate If** (bullets)
            """
        
        response = model.generate_content([prompt, pil_img])
        return response.text
    except Exception as e:
        return f"Gemini error (model {model_name}): {e}"

def fetch_available_gemini_models():
    api_key = get_gemini_api_key()
    if not api_key:
        return []
    try:
        global genai
        if genai is None:
            import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in getattr(m, 'supported_generation_methods', []):
                name = getattr(m, 'name', '')
                if name.startswith('models/'):
                    name = name[len('models/'):]
                if name:
                    models.append(name)
        return sorted(set(models))
    except Exception:
        return []


# --- Main App ---
def main():
    st.set_page_config(page_title="AI Plant Disease Agent", page_icon="🌿", layout="wide")
    
    current_lang = st.session_state.get("ui_lang", "en")
    lang_options = list(LANG_OPTIONS.keys())
    default_lang_label = next((k for k, v in LANG_OPTIONS.items() if v == current_lang), "English")
    default_lang_index = lang_options.index(default_lang_label) if default_lang_label in lang_options else 0
    lang_label = st.sidebar.selectbox(
        tr(current_lang, "language"),
        lang_options,
        index=default_lang_index,
        key="lang_label",
    )
    lang = LANG_OPTIONS.get(st.session_state.get("lang_label", lang_label), "en")
    st.session_state["ui_lang"] = lang

    st.title(tr(lang, "app_title"))
    st.markdown(f"### {tr(lang, 'app_subtitle')}")
    
    # Initialize session state for results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'gemini_models' not in st.session_state:
        st.session_state.gemini_models = []
    
    # Sidebar
    st.sidebar.title(tr(lang, "settings"))
    use_gemini = st.sidebar.toggle(tr(lang, "use_gemini"), value=False)
    guidance_mode = "short"
    model_name = ""
    if use_gemini:
        guidance_mode_label = st.sidebar.selectbox(tr(lang, "guidance_mode"), [tr(lang, "mode_short"), tr(lang, "mode_detailed")], index=0)
        guidance_mode = "detailed" if guidance_mode_label == tr(lang, "mode_detailed") else "short"
        if not get_gemini_api_key():
            st.sidebar.warning(tr(lang, "gemini_not_configured"))
            with st.sidebar.expander(tr(lang, "advanced"), expanded=False):
                st.caption("Option 1: set .streamlit/secrets.toml (recommended)")
                st.caption("Option 2: temporary key (not saved)")
                st.text_input("Temporary Gemini API Key", type="password", key="temp_gemini_api_key")

        if not st.session_state.gemini_models and get_gemini_api_key():
            st.session_state.gemini_models = fetch_available_gemini_models()
        if st.sidebar.button(tr(lang, "load_models")):
            st.session_state.gemini_models = fetch_available_gemini_models()
        auto_model = get_preferred_gemini_model(st.session_state.gemini_models) if st.session_state.gemini_models else ""
        model_name = auto_model
        if get_gemini_api_key():
            with st.sidebar.expander(tr(lang, "advanced"), expanded=False):
                if st.session_state.gemini_models:
                    model_name = st.selectbox(tr(lang, "model"), st.session_state.gemini_models, index=st.session_state.gemini_models.index(auto_model) if auto_model in st.session_state.gemini_models else 0, key="gemini_model")
                else:
                    model_name = st.text_input(tr(lang, "model"), value=model_name, key="gemini_model_text")

    conf_threshold = 0.4
    min_disease_conf = 0.6
    max_leaves = 5
    crop_padding = 0.1
    plant_filter = "Auto"
    
    st.sidebar.markdown("---")
    st.sidebar.info(tr(lang, "how_it_works"))
    
    # Model loading
    leaf_model, disease_model, device = load_models()
    if leaf_model is None or disease_model is None:
        st.stop()
        
    # File Uploader
    uploaded_file = st.file_uploader(tr(lang, "upload"), type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.subheader(tr(lang, "uploaded"))
        st.image(image, use_container_width=True)
        
        if st.button(tr(lang, "analyze")):
            with st.spinner("Analyzing..."):
                # 1. Leaf Detection
                results = leaf_model(img_bgr, conf=conf_threshold, verbose=False)
                
                candidates = []
                h, w, _ = img_bgr.shape
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    result_obj = results[0]
                    boxes = result_obj.boxes.xyxy.cpu().numpy()
                    leaf_confs = result_obj.boxes.conf.cpu().numpy()
                    order = np.argsort(leaf_confs)[::-1][:max_leaves]
                    for idx in order:
                        x1, y1, x2, y2 = map(int, boxes[int(idx)])
                        pad_x = int((x2 - x1) * crop_padding)
                        pad_y = int((y2 - y1) * crop_padding)
                        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        x2, y2 = min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)
                        crop = img_bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        probs = classify_disease(crop, disease_model, device)
                        topk = get_topk_predictions(probs, plant_filter=plant_filter, k=3)
                        if not topk:
                            continue
                        top1_name, raw_p, p = topk[0]
                        leaf_p = float(leaf_confs[int(idx)])
                        combined = leaf_p * float(p)
                        candidates.append({
                            "crop": crop,
                            "topk": topk,
                            "top1": top1_name,
                            "raw_conf": float(raw_p),
                            "score": float(p),
                            "leaf_conf": leaf_p,
                            "combined": combined
                        })
                else:
                    probs = classify_disease(img_bgr, disease_model, device)
                    topk = get_topk_predictions(probs, plant_filter=plant_filter, k=3)
                    if topk:
                        top1_name, raw_p, p = topk[0]
                        candidates.append({
                            "crop": img_bgr,
                            "topk": topk,
                            "top1": top1_name,
                            "raw_conf": float(raw_p),
                            "score": float(p),
                            "leaf_conf": 1.0,
                            "combined": float(p)
                        })

                if not candidates:
                    st.error(tr(lang, "classify_fail"))
                    st.session_state.analysis_results = None
                    return

                best = max(candidates, key=lambda x: x["combined"])
                disease_class = best["top1"]
                raw_conf = best["raw_conf"]
                topk = best["topk"]
                cropped_leaf = best["crop"]

                parts = disease_class.split('___', 1)
                plant_type = parts[0].replace('_', ' ')
                condition = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"

                low_confidence = float(best["score"]) < float(min_disease_conf)

                current_detection = {
                    'timestamp': datetime.now().isoformat(),
                    'plant_type': plant_type,
                    'condition': condition if not low_confidence else f"Uncertain: {condition}",
                    'confidence': f"{raw_conf:.4f}",
                }
                save_to_csv(current_detection)

                cropped_leaf_rgb = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                if use_gemini and model_name and get_gemini_api_key():
                    advice = get_genai_recommendations(cropped_leaf_rgb, plant_type, condition, model_name, guidance_mode, lang)
                else:
                    advice = get_offline_guidance_short(plant_type, condition, lang)

                st.session_state.analysis_results = {
                    'plant_type': plant_type,
                    'condition': condition,
                    'confidence': float(best["score"]),
                    'raw_confidence': raw_conf,
                    'cropped_leaf': cropped_leaf_rgb,
                    'advice': advice,
                    'topk': topk,
                    'low_confidence': low_confidence,
                    'plant_filter': plant_filter,
                    'leaf_confidence': float(best["leaf_conf"])
                }

        # Display Results from Session State
        if st.session_state.analysis_results:
            res = st.session_state.analysis_results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(tr(lang, "classification"))
                st.image(res['cropped_leaf'], caption=tr(lang, "detected_leaf"), use_container_width=True)
                plant_disp, cond_disp = translate_pair(res['plant_type'], res['condition'], lang)
                st.metric(tr(lang, "label_plant"), plant_disp)
                st.metric(tr(lang, "label_condition"), cond_disp)
                st.metric(tr(lang, "label_confidence"), f"{res.get('confidence', 0.0)*100:.2f}%")
                if res.get('low_confidence'):
                    st.warning(tr(lang, "low_conf"))
                if res.get('plant_filter') and res['plant_filter'] != "Auto":
                    pf = translate_plant(res['plant_filter'], lang)
                    st.caption(f"{('पौधा फ़िल्टर लागू:' if lang == 'hi' else 'Plant type filter applied:')} {pf}")
                if res.get('topk'):
                    st.write(tr(lang, "top_predictions"))
                    for name, raw_p, _ in res['topk']:
                        parts = name.split('___', 1)
                        plant = parts[0].replace('_', ' ')
                        cond = parts[1].replace('_', ' ') if len(parts) > 1 else name
                        plant_t, cond_t = translate_pair(plant, cond, lang)
                        st.write(f"- {plant_t} — {cond_t}: {raw_p*100:.2f}%")
                
                if "healthy" in res['condition'].lower():
                    if lang == "hi":
                        st.success(f"अच्छी खबर! आपका {plant_disp} स्वस्थ लग रहा है।")
                    else:
                        st.success(f"Great news! Your {plant_disp} looks healthy.")
                else:
                    if lang == "hi":
                        st.error(f"समस्या मिली: {cond_disp}")
                    else:
                        st.error(f"Issue detected: {cond_disp}")
            
            with col2:
                st.subheader(f"🤖 {tr(lang, 'ai_guidance')}")
                st.markdown(res['advice'])
    else:
        # Clear results if no file is uploaded
        st.session_state.analysis_results = None

if __name__ == "__main__":
    main()
