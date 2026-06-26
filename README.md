# Speech Disorder Classification Using Hybrid Deep Neural Networks and Concatenated Feature Embedding

## 📌 Overview
This project presents a hybrid deep learning framework for the automatic classification of dysarthric and non-dysarthric speech. The proposed model combines Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) networks to learn both spatial and temporal characteristics from speech signals represented as Mel-spectrograms.

The system aims to support early diagnosis of dysarthria by providing an automated, accurate, and efficient speech classification solution.

---

## 🚀 Features
- Automatic speech disorder detection
- Mel-spectrogram based feature representation
- CNN for spatial feature extraction
- Parallel LSTM and GRU for temporal modeling
- Feature concatenation for improved classification
- Binary classification (Dysarthric / Non-Dysarthric)
- Performance evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC Curve

---

## 🏗️ Project Workflow

```
Input (.wav)
      │
      ▼
Preprocessing
      │
      ▼
Mel-Spectrogram Generation
      │
      ▼
Feature Normalization & Data Splitting
      │
      ▼
CNN Feature Extraction
      │
      ▼
Reshape to Sequence
      │
 ┌────┴────┐
 ▼         ▼
LSTM      GRU
 └────┬────┘
      ▼
Feature Concatenation
      ▼
Dense Layer
      ▼
Softmax Classification
(Dysarthric / Non-Dysarthric)
```

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## 📂 Dataset

- Audio Format: `.wav`
- Feature Representation: Mel-Spectrogram
- Classification:
  - Dysarthric Speech
  - Non-Dysarthric Speech

Dataset Source:
https://data.mendeley.com/datasets/3mhnr7frht/1

---

## 🧠 Model Architecture

- CNN (Conv2D + ReLU + MaxPooling)
- LSTM
- GRU
- Feature Concatenation
- Dense Layer
- Softmax Output Layer

---

## 📊 Performance

| Model | Accuracy |
|--------|----------|
| CNN + GRU | 81% |
| CNN + LSTM | 85% |
| Hybrid CNN + LSTM + GRU | 94% |

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

## 📁 Project Structure

```
├── Dataset/
├── Notebooks/
├── Models/
├── Results/
├── Figures/
├── requirements.txt
├── README.md
└── Speech_Disorder_Classification.ipynb
```

---

## ▶️ Installation

```bash
git clone <repository-url>

cd <repository-name>

pip install -r requirements.txt
```

---

## ▶️ Run the Project

Open the Jupyter Notebook or Google Colab notebook and execute all cells sequentially.

---

## 📈 Results

The proposed hybrid CNN-LSTM-GRU model demonstrated superior performance compared to individual CNN-LSTM and CNN-GRU architectures. By combining spatial feature extraction with temporal sequence learning, the model achieved improved classification accuracy and robustness for dysarthria detection.

---

## 🔮 Future Work

- Multi-class dysarthria severity classification
- Attention mechanisms
- Transformer-based speech models
- Explainable AI (XAI)
- Cross-dataset validation
- Real-time speech disorder detection

---

## 👨‍💻 Author

**H J Chaithanya Prasad**

Integrated M.Tech Computer Science and Engineering

Vellore Institute of Technology, Vellore

---

## 📄 License

This project is intended for academic and research purposes.
