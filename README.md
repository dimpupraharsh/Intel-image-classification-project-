
# 🌄 Multi-Class Image Classification Using CNNs and Transfer Learning (VGG16)

Welcome to our deep learning project where we explore the effectiveness of custom CNNs and transfer learning with VGG16 for classifying natural and urban scenes. We compare model performance, tackle overfitting, and demonstrate the impact of strategic fine-tuning.

---

## 📌 Project Overview

This project focuses on multi-class image classification using both a custom Convolutional Neural Network (CNN) and the VGG16 pre-trained model. We aim to:

- Build a baseline CNN from scratch.
- Improve performance using transfer learning with VGG16.
- Apply fine-tuning and hyperparameter optimization.
- Evaluate results using accuracy metrics and confusion matrices.

---

## 🧠 Key Features

- 📁 **Dataset**: Over 14,000 images, categorized into 6 classes — buildings, forests, glaciers, mountains, seas, and streets. Image size: 150x150 pixels.
- 🧪 **Custom CNN**: Baseline model trained from scratch.
- 🔁 **Transfer Learning**: VGG16 model with frozen layers and later selective unfreezing for fine-tuning.
- ⚙️ **Optimizations**: Adaptive learning rates, early stopping, and confusion matrix analysis.
- 📊 **Performance**: Accuracy improved from ~17% (custom model on test set) to **92%** after fine-tuning VGG16.

---

## 📂 Dataset Details

| Feature             | Description |
|---------------------|-------------|
| **Total Images**    | 14,000+     |
| **Classes**         | Buildings, Forests, Glaciers, Mountains, Seas, Streets |
| **Image Resolution**| 150x150 pixels |
| **Data Split**      | Train / Test / Prediction sets |

---

## 🚀 Models Compared

### 🧱 Base CNN
- 86.09% training accuracy, 85.31% validation accuracy.
- Severe overfitting → 17.13% test accuracy.

### 🧠 VGG16 (Transfer Learning)
- Initial training with frozen layers → 85.77% validation accuracy.
- After fine-tuning (unfreezing deeper layers) → **92% test accuracy**.

---

## 📈 Results Summary

| Model       | Train Acc | Val Acc | Test Acc | Overfitting |
|-------------|-----------|---------|----------|-------------|
| Base CNN    | 86.09%    | 85.31%  | 17.13%   | High        |
| VGG16 (TL)  | 99.15%    | 92.04%  | 92%      | Low         |

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Matplotlib & Seaborn (Visualization)  
- Scikit-learn (Metrics & Evaluation)  
- Google Colab (Training Environment)

---

## 📊 Evaluation

- 📉 **Loss & Accuracy Curves**: Monitored for overfitting and convergence.
- ✅ **Confusion Matrix**: Used for detailed class-wise performance insights.
- 🧪 **Best Epoch**: Identified optimal stopping points based on validation performance.

---

## 🔍 Limitations & Considerations

- Overfitting in custom CNN due to limited data.
- Domain mismatch between ImageNet and our dataset necessitated fine-tuning.
- Resource demands of VGG16.
- Ethical considerations: dataset fairness, privacy, and explainability.

---

## 📚 References

- Simonyan & Zisserman (2014): VGG16 Paper  
- ImageNet Challenge Overview (Russakovsky et al.)  
- TensorFlow Docs: https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16  
- Keras Docs: https://keras.io/api/applications/vgg/#vgg16-function  

---

## 📎 Authors
- Medi Praharsh Vijay  

---

## 🔗 Colab Notebook

📘 Run the notebook here:  
https://colab.research.google.com/drive/1or_iiRKke1SBUwxUokgN96OZVXXMnlrT#scrollTo=81445993
