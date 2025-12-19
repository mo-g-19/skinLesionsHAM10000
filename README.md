# Skin Lesion Classification with ResNet50 and ResNet50-SE (HAM10000 Dataset)

This project implements and compares two deep learning architectures for multiclass skin lesion classification using the **HAM10000** dermoscopic image dataset:

1. **Baseline ResNet50**
2. **ResNet50-SE** — a modified ResNet50 architecture incorporating a **Squeeze-and-Excitation (SE)** attention block

The goal is to evaluate whether channel-wise attention improves classification performance for challenging medical imaging tasks such as melanoma detection.

---

## Overview

Skin cancer is the most common cancer in the United States, and early detection is critical for improving survival outcomes. Deep learning models, particularly convolutional neural networks (CNNs), have shown strong performance in automated skin lesion classification.

This project applies two CNN architectures to classify the seven HAM10000 lesion categories:

- akiec — Actinic keratoses / intraepithelial carcinoma  
- bcc — Basal cell carcinoma  
- bkl — Benign keratosis-like lesions  
- df — Dermatofibroma  
- mel — Melanoma  
- nv — Melanocytic nevi  
- vasc — Vascular lesions

---

## Models

### **1. Baseline ResNet50**
- Pretrained on ImageNet  
- Early layers frozen, later layers fine-tuned  
- Classification head: GlobalAveragePooling → Dropout → Dense(7)  

### **2. ResNet50-SE (Channel Attention Model)**
- Same backbone as ResNet50  
- Adds a **Squeeze-and-Excitation (SE) block** after feature extraction  
- Learns channel-wise feature importance  
- Improves representation with minimal computational overhead  

---

## Results

| Model             | Test Accuracy | Training Time | Notes |
|------------------|----------------|----------------|-------|
| ResNet50 (Base)  | ~0.755         | ~847 sec       | Strong performance on majority class (nv) |
| ResNet50-SE      | ~0.778         | ~847 sec       | Better balance across minority classes |

### Confusion Matrices
See:
- `results/confusion_matrix_resnet50_base.png`
- `results/confusion_matrix_resnet50_channel.png`

Key observations:
- High accuracy on the majority class (**nv**)  
- Common melanoma misclassifications (mel → nv, mel → bkl)  
- SE model reduces some confusion and improves classification stability  

---

## Dataset Preparation

1. Download the HAM10000 dataset.  
2. Combine image parts and join with metadata.  
3. Create stratified train/val/test splits (70/15/15).  
4. Preprocess:
   - Resize to 224×224  
   - Normalize pixel values  
   - Apply augmentation (flip, rotate, zoom)  

---

## Training Details

- Framework: **TensorFlow / Keras**  
- Hardware: **Google Colab GPU**  
- Optimizer: Adam  
- Learning rates:  
  - 1e-4 (frozen stage)   
- Batch size: 32  
- Epochs: 15  

Both models used:
- Early stopping  
- Checkpointing  
- Identical training pipeline for fair comparison  

---

## Repository Structure

```text
skinLesionsHAM10000/
├── notebooks/
│   └── skin_lesion_classification.ipynb
│
├── models/
│   ├── resnet50_lr4_best.keras
│   └── resnet50_se_lr4_best.keras
│
├── results/
│   ├── confusion_matrix_resnet50_base.png
│   ├── confusion_matrix_resnet50_channel.png
│   └── metrics.txt
│
└── README.md
```



---

## How to Run
1) You must download the dataet separately from Kaggle:

**Dataset link:**  
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

After downloading, you should have a ZIP file containing at least:
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/

2) After downloading it from Kaggle, upload it to GoogleDrive.
In GoogleDrive, the final path should be:
  drive/MyDrive/archive.zip

3) Open a notebook in Google Colab:
- click File -> Open Notebook
- Selec GitHub tab
- Enter this repository URL: mo-g-19/skinLesionsHAM10000
- Click on notebooks/main_experiments.ipynb to open it

in the first bubble, clone the repository, run:
  !git clone https://github.com/mo-g-19/skinLesionsHAM10000
  %cd /content/skinLesionsHAM10000/
  !ls

in the second bubble, install required Python packages, run:
  pip install -r requirements.txt

in the third bubble, run:
  notebooks/main_experiments.ipynb

This one call should:
- Mount Google Drive
- Extract the dataset ZIP from Drive
- Run prepare_ham10000_splits.py
- create persistent train/bal/test folders on Drive
- Set DATA_ROOT for the rest of the notebook

---

## Troubleshooting
Problem: "archive.zip not found in Google Drive"
Solution:
- Double-check you've uploaded archive.zip to the root of "My Drive"
- The path should be exactly: "MyDrive/archive.zip"
- Try refreshing your Drive or re-uploading the file

## Requirements

- Google Colab (recommended)
- Python 3.9+
- A Google account (for Drive storage)

All Python dependencies are listed in `requirements.txt`.

---

## Dataset: HAM10000

The HAM10000 dataset is **not included** in this repository due to size and licensing constraints.




## References

He et al., Deep Residual Learning for Image Recognition, 2015
Hu et al., Squeeze-and-Excitation Networks, 2017
Vaswani et al., Attention Is All You Need, 2017
Esteva et al., Dermatologist-level classification of skin cancer, Nature, 2017
Tschandl et al., The HAM10000 Dataset, 2018

(Full citations included in the project report.)

## License
This project is for educational and resarch purposes. Please cite the original HAM100000 dataset if you use this code in your research