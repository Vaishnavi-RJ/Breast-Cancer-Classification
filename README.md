# 🧠 Breast Cancer Tumor Classification using Neural Networks

This project uses a neural network built with **TensorFlow** and **Keras** to classify tumors as **benign** or **malignant** using the **Wisconsin Breast Cancer Dataset**.

## 📊 Dataset

The [Wisconsin Breast Cancer Dataset (WBCD)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) contains 30 numerical features computed from digitized images of breast mass cell nuclei.

- Features: radius, texture, perimeter, area, smoothness, etc.
- Labels: `M` (Malignant), `B` (Benign)

## 🚀 Project Workflow

1. **Data Loading & Exploration**
   - Loaded dataset and checked for missing values
   - Analyzed class distribution and feature correlations

2. **Preprocessing**
   - Converted labels to binary (0 = Benign, 1 = Malignant)
   - Applied feature scaling using `StandardScaler`
   - Split dataset into training and test sets

3. **Model Building**
   - Constructed a **feedforward neural network** using Keras
   - Layers: Input → Dense (ReLU) → Dense (ReLU) → Output (Sigmoid)
   - Optimizer: Adam | Loss: Binary Crossentropy

4. **Model Evaluation**
   - Achieved **95% accuracy** on the test set
   - Evaluated with confusion matrix, precision, recall, and F1-score

## 🧪 Example Architecture

```python
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(30,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
📈 Results
Accuracy: 95%

Precision / Recall / F1-score: Evaluated using sklearn.metrics

Plotted training vs validation loss and accuracy

🛠️ Technologies Used
Python

Pandas, NumPy

TensorFlow & Keras

Scikit-learn (data preprocessing & evaluation)

Matplotlib / Seaborn (visualizations)
