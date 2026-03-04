# 🐶 Dog Breed Classification using Transfer Learning

A deep learning project that classifies 120 dog breeds from images using **Transfer Learning** with InceptionV3 pre-trained on ImageNet.

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Task** | Multi-class image classification |
| **Classes** | 120 dog breeds |
| **Dataset Size** | ~10,000 labeled images |
| **Model** | InceptionV3 + custom head |
| **Framework** | TensorFlow / Keras |
| **Technique** | Transfer Learning |

## 🗂️ Repository Structure

```
dog-breed-classification/
│
├── dog_breed_classification.ipynb   # Main notebook (full pipeline)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── training_history.png             # Loss/AUC plots (generated after training)
```

## 📊 Dataset

Download the dataset here: [Dog Breed Identification Dataset](https://drive.google.com/file/d/14of-v7y9Q95fqBvOXfTrEg2C2LZ-aq6l/view?usp=sharing)

After downloading, place `dog-breed-identification.zip` in the project root. The notebook will extract it automatically.

**Structure after extraction:**
```
train/          ← labeled training images (.jpg)
labels.csv      ← image ID + breed label
```

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dog-breed-classification.git
cd dog-breed-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Place `dog-breed-identification.zip` in the project root directory.

### 4. Run the notebook
```bash
jupyter notebook dog_breed_classification.ipynb
```

## 🧠 Model Architecture

```
InceptionV3 (pre-trained, frozen up to 'mixed7' layer)
        ↓
    Flatten
        ↓
  Dense(256, ReLU)
        ↓
  BatchNormalization
        ↓
  Dense(256, ReLU)
        ↓
   Dropout(0.3)
        ↓
  BatchNormalization
        ↓
  Dense(120, Softmax)   ← output: 120 breed probabilities
```

## 🔄 Pipeline

1. **Data Loading** — Read images from disk with `labels.csv`
2. **EDA** — Visualize class distribution and sample images
3. **Label Encoding** — Encode 120 breed strings to integers
4. **Train/Val Split** — 85% training / 15% validation
5. **Augmentation** — Albumentations (flip, dropout, brightness, gamma)
6. **Input Pipeline** — `tf.data.Dataset` with parallel loading & prefetching
7. **Model Building** — InceptionV3 backbone + custom dense head
8. **Training** — Adam optimizer, early stopping, LR scheduler
9. **Evaluation** — AUC metric, loss/AUC curves
10. **Inference** — Predict top-3 breeds for any image

## 📈 Training Callbacks

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stops if val AUC doesn't improve for 3 epochs |
| `ReduceLROnPlateau` | Halves LR if val loss plateaus for 2 epochs |
| `AUCThresholdCallback` | Stops training if val AUC > 0.99 |

## 🖼️ Sample Results

The model predicts the top-3 most likely breeds with confidence scores:

```
Actual breed: golden_retriever
Top 3 Predictions:
  1. golden_retriever: 94.23%
  2. labrador_retriever: 3.11%
  3. flat-coated_retriever: 1.05%
```

## 🛠️ Technologies Used

- Python 3.10
- TensorFlow 2.x / Keras
- OpenCV
- Albumentations
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

## 📚 References

- [Transfer Learning — GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/dog-breed-classification-using-transfer-learning/)
- [InceptionV3 Paper](https://arxiv.org/abs/1512.00567)
- [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

## 📄 License

This project is for educational and portfolio purposes.
