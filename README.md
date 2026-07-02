# Arabic Handwritten Characters Classifier

A CNN-based image classifier trained to recognize all 28 Arabic handwritten characters using the [AHCD dataset](https://www.kaggle.com/datasets/mloey1/ahcd1). The project covers the full pipeline — from data exploration to training, transfer learning, and evaluation.

---

## Results

| Model | Test Accuracy |
|---|---|
| CNN from Scratch | 94.14% |
| CNN + Transfer Learning | 94.49% |

Both models were evaluated using confusion matrices across all 28 character classes.

---

## Approach

**1. Data Loading & Exploration**
Loaded the AHCD dataset and visualized sample characters across all 28 classes to understand class distribution and image quality before training.

**2. CNN from Scratch**
Built a custom CNN architecture using:

- Conv2D layers for feature extraction
- MaxPooling for spatial downsampling
- Batch Normalization for training stability
- Dropout for regularization
- Global Max Pooling before the classification head

**3. Transfer Learning**
Saved the trained model, applied data augmentation, then reloaded it and froze the lower convolutional layers. Retrained only the higher layers on the augmented data, pushing test accuracy from 94.14% to 94.49%.

**4. Evaluation**
Both models were evaluated using confusion matrices to inspect per-class performance across all 28 Arabic characters.

---

## Tech Stack

- Python, TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn
- AHCD Dataset

---

## Project Structure

``` python

├── arabic_classifier.ipynb      # Full training pipeline
├── evaluation.ipynb             # Evaluation notebook
├── model_scratch.keras          # Saved base CNN model
├── model_transfer.keras         # Saved transfer learning model
└── README.md
```

---

## References

- [AHCD Dataset on Kaggle](https://www.kaggle.com/datasets/mloey1/ahcd1)
- [GitHub Repository](https://github.com/Sam50x/arabic-handwritten-characters-classifier)
