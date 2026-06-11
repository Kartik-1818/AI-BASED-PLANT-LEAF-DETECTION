4. **(Optional) Add Gemini API key** for enhanced AI guidance:
```bash
   cp .streamlit/secrets.example.toml .streamlit/secrets.toml
   # then add your key inside secrets.toml
```
   Free keys at [aistudio.google.com](https://aistudio.google.com/)

5. **Launch:**
```bash
   streamlit run Plant_Disease_Agent.py
```

---

## ⚠️ Honest Limitations

The CNN is trained on the PlantVillage benchmark dataset — it knows 38 disease classes across 14 species. If you photograph a wheat or rice leaf, it won't know what to do. That's a dataset limitation, not an architectural one — the pipeline is built to scale.

**What's next:**
- [ ] Field-collected images for regional Indian crops (wheat, rice, sugarcane)
- [ ] User feedback loop to continuously improve predictions
- [ ] Mobile-optimized version for low-end Android devices
- [ ] Streamlit Cloud public deployment

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [Ultralytics YOLO](https://ultralytics.com/)
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://aistudio.google.com/)

---