
## 🧠 Brain Tumor Detection Gradio App

### 🚀 Getting Started

To run this Gradio-based app, follow the steps below:

### 📁 Dataset

* **Dataset Used:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
* This dataset includes 4 classes: `glioma`, `meningioma`, `pituitary`, and `no tumor`.

### 🗂️ Data Preparation

Before training the model:

1. **Download the dataset** from the Kaggle link above.
2. **Combine the training and testing files** into a **single folder per class**.
3. Your final folder structure should look like this:

```
BrainTumorDataset/
├── glioma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── meningioma/
│   ├── image1.jpg
│   └── ...
├── pituitary/
│   ├── image1.jpg
│   └── ...
└── no_tumor/
    ├── image1.jpg
    └── ...
```

### ⚙️ Model Training

* Train your model on the cleaned and combined dataset before running the app.
* The trained model will be used by the Gradio app for predictions.

### 🖼️ Gradio App Note

* The app currently supports **only 1–2 images** at a time.
* It may crash with more images — this will be optimized in future updates.

---

