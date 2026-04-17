# 🧠 Brain Imaging Analysis Studio

> Clinical Decision Support System for **CT Hemorrhage Detection** and **MRI Tumor Segmentation** with Explainable AI (XAI)

---

## 🚀 Overview

Brain Imaging Analysis Studio is an integrated AI-based system that processes both **CT** and **MRI** brain scans within a unified pipeline.

It helps in:
- 🩸 Detecting hemorrhage in CT scans  
- 🧬 Segmenting tumors in MRI scans  
- 🔍 Providing visual explanations (XAI)  
- 📊 Estimating severity and risk  
- 🧾 Generating patient-friendly insights  

---

## 🧩 Features

- 🧠 Multi-Modal Analysis (CT + MRI)  
- 📊 Severity & Risk Visualization  
- 🔍 Explainable AI (Grad-CAM & Overlays)  
- ⚖️ Class Imbalance Handling  
- 📈 Model Comparison & Optimization  
- 🖥️ Clinical Dashboard UI  

---

## 🏗️ System Architecture

```text
Input → Preprocessing → Model → Explainability → Severity → Report
```

### CT Pipeline
- HU Windowing → DenseNet121 → Grad-CAM → Severity  

### MRI Pipeline
- Slice Extraction → 2D U-Net → Segmentation → Tumor %  

---

## 🗂️ Datasets

### 📌 RSNA Intracranial Hemorrhage
- ~750K filtered CT images  
- Binary classification  

### 📌 BraTS 2021
- 1251 patients  
- Modalities: FLAIR, T1ce, T2  

---

## ⚙️ Tech Stack

- 🐍 Python  
- 🧠 TensorFlow / Keras  
- 🔬 OpenCV, NumPy  
- 📊 Matplotlib  
- 🖥️ Streamlit  
- 🎨 HTML + CSS  

---

## 📊 Results

### 🩸 CT Hemorrhage Detection
- Accuracy: **85%**  
- Recall: **0.84**  
- AUC: **0.92**  
- Optimal Threshold: **0.4**  

### 🧬 MRI Tumor Segmentation
- Dice Score: **~0.82**  
- IoU: **~0.91**  
- Best Epoch: **9**  

---

## 🧪 Model Comparison

| Model        | Accuracy | Sensitivity | Specificity |
|--------------|---------:|------------:|------------:|
| CNN          | 0.82     | 0.35        | 0.90        |
| ResNet50     | 0.85     | 0.40        | 1.00        |
| DenseNet121  | **0.89** | **0.51**    | 0.95        |

✔ DenseNet121 selected for best balanced performance  

---

## 🔍 Explainability

- Grad-CAM highlights important regions in CT scans  
- MRI overlay shows tumor localization  
- Improves model transparency and trust  

---

## 🖥️ UI Features

- 📊 Dashboard-style layout  
- 📈 Risk & severity indicators  
- 🧠 Image visualization panels  
- 🧾 Patient-friendly reports  

---

## 📁 Project Structure

```text
project/
│── ct_model/
│── mri_model/
│── preprocessing/
│── ui/
│── results/
│── app.py
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/your-repo-name
cd project
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Future Work

- Extend to X-ray, chest, and retina imaging
- Integrate LLM-based report generation
- Deploy on cloud platforms (AWS / GCP)
- Add mobile-friendly interface

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.  
It is **NOT a substitute for professional medical advice, diagnosis, or treatment**.

---

## 👩‍💻 Author

**Rishitha Konidena**  
Computer Science Engineering  
AI & Healthcare Enthusiast

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
