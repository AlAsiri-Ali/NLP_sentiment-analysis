# Android Galaxy Sentiment Analysis

**🎯 Natural Language Processing project for sentiment classification of Android Galaxy product reviews**

> Built with traditional machine learning and deep learning approaches to classify customer sentiment from Reddit reviews.

## Overview

This project analyzes customer sentiment toward Android Galaxy devices using Natural Language Processing techniques. The system compares traditional machine learning models with deep learning approaches to classify Reddit reviews as positive, negative, or neutral.

### Key Features
- **Multiple AI Models**: Logistic Regression, SVM, Random Forest, and Bidirectional LSTM
- **Real Dataset**: 1,000 authentic Android Galaxy reviews from Reddit
- **High Accuracy**: Achieves 77.5% accuracy with Random Forest model
- **Complete Pipeline**: From raw text preprocessing to final classification

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python main.py
```

### Quick Demo
```bash
python demo.py
```

---

## Project Structure

```
android-galaxy-sentiment/
├── main.py                    # Main analysis pipeline
├── demo.py                    # Interactive demonstration
├── quick_test.py              # Quick functionality test
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
│
├── data/
│   └── sample_data.csv        # 1,000 Android Galaxy reviews
│
├── notebooks/
│   └── Project_EMAI631.ipynb  # Research and development notebook
│
└── src/
    ├── __init__.py            # Package initialization
    ├── data_preprocessing.py  # Data cleaning & preparation
    ├── text_representation.py # Feature extraction (TF-IDF, Word2Vec)
    ├── traditional_models.py  # ML models (LR, SVM, RF)
    ├── deep_learning_model.py # LSTM implementation
    └── utils.py               # Utility functions
```

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **77.5%** | 0.78 | 0.77 | 0.77 |
| **Logistic Regression** | 75.2% | 0.75 | 0.75 | 0.75 |
| **SVM (Linear)** | 74.1% | 0.74 | 0.74 | 0.74 |
| **Bidirectional LSTM** | 73.8% | 0.74 | 0.73 | 0.73 |

### Dataset Information
- **Total Reviews**: 1,000 Android Galaxy reviews from Reddit
- **Classes**: Positive, Negative, Neutral
- **Class Distribution**: 75.2% Negative, 17.3% Positive, 7.5% Neutral
- **Data Balancing**: Applied SMOTE upsampling

### Key Insights
- **Best Model**: Random Forest achieves highest accuracy (77.5%)
- **Fastest Training**: Logistic Regression trains in ~5 seconds
- **Deep Learning**: LSTM provides comparable results to traditional methods
- **Feature Analysis**: Words like "amazing", "terrible", "battery life" are most predictive

---

## Usage Examples

### Basic Usage
```python
from src.traditional_models import train_traditional_models
from src.data_preprocessing import load_and_preprocess_data

# Load and preprocess data
data = load_and_preprocess_data('data/sample_data.csv')

# Train all models
results = train_traditional_models(data)
print(f"Best accuracy: {max(results.values()):.3f}")
```

### Deep Learning Model
```python
from src.deep_learning_model import train_lstm_model

# Train LSTM model
model, history, encoder, tokenizer = train_lstm_model(data)

# Make prediction
new_review = "This Galaxy phone has amazing camera quality!"
prediction = model.predict(new_review)
```

---

## Technical Details

### Text Preprocessing
- HTML tag removal and text cleaning
- Tokenization and lemmatization using NLTK
- Stop word removal and lowercasing
- SMOTE upsampling for class balance

### Feature Extraction
- **Bag of Words**: Basic word frequency vectors
- **TF-IDF**: Term frequency-inverse document frequency
- **Word2Vec**: Pre-trained word embeddings

### Models Implemented
- **Logistic Regression**: Linear classification baseline
- **Support Vector Machine**: Non-linear classification with RBF kernel
- **Random Forest**: Ensemble method with 100 trees
- **Bidirectional LSTM**: Deep learning with 64 units and dropout

---

## Requirements

```
pandas>=1.5.3
numpy>=1.23.5
scikit-learn>=1.3.2
nltk>=3.8.1
tensorflow>=2.15.0
gensim>=4.3.2
matplotlib>=3.7.1
seaborn>=0.12.2
imbalanced-learn>=0.11.0
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

### Academic Use
This project was developed for the Natural Language Processing course (EMAI631) as part of a Master's program. If you use this work in academic research, please provide appropriate citation.

---

## Acknowledgments

- **Course**: Natural Language Processing (EMAI631)
- **Data Source**: Reddit discussions on Android Galaxy devices
- **Technologies**: Python, scikit-learn, TensorFlow, NLTK

---

*Natural Language Processing Project | Sentiment Analysis | Master's Program 2025*