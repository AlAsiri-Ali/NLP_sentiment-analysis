"""
Android Galaxy Sentiment Analysis - Interactive Demo
Professional NLP pipeline for sentiment classification using multiple ML approaches
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from collections import Counter
import re

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase and strip
    text = text.lower().strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():
    print("=" * 60)
    print("ANDROID GALAXY SENTIMENT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("Professional NLP Pipeline: Traditional ML + Deep Learning")
    print("Analyzing Android Galaxy Product Reviews from Reddit")
    print("=" * 60)
    
    # 1. Data Loading and Exploration
    print("\n📊 STEP 1: DATA LOADING AND EXPLORATION")
    print("-" * 40)
    
    df = pd.read_csv('data/sample_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print(f"\nSample reviews:")
    for i, (text, label) in enumerate(zip(df['comment_text'].head(3), df['label'].head(3))):
        print(f"  {i+1}. [{label.upper()}] {text[:80]}...")
    
    # 2. Text Preprocessing
    print("\n🔤 STEP 2: TEXT PREPROCESSING")
    print("-" * 40)
    
    print("Applying text cleaning...")
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    
    # Show before/after example
    idx = 0
    print(f"\nBefore: {df['comment_text'].iloc[idx]}")
    print(f"After:  {df['cleaned_text'].iloc[idx]}")
    
    # Text statistics
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    
    print(f"\nText Statistics:")
    print(f"  Average text length: {df['text_length'].mean():.0f} characters")
    print(f"  Average word count: {df['word_count'].mean():.0f} words")
    print(f"  Max text length: {df['text_length'].max()} characters")
    print(f"  Min text length: {df['text_length'].min()} characters")
    
    # 3. Data Splitting
    print("\n📊 STEP 3: DATA PREPARATION")
    print("-" * 40)
    
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training label distribution: {dict(y_train.value_counts())}")
    
    # 4. Feature Extraction
    print("\n🔧 STEP 4: FEATURE EXTRACTION")
    print("-" * 40)
    
    # TF-IDF Vectorization
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    
    # Show top features
    feature_names = tfidf.get_feature_names_out()
    print(f"Sample features: {list(feature_names[:10])}")
    
    # 5. Traditional Machine Learning Models
    print("\n🤖 STEP 5: TRADITIONAL MACHINE LEARNING")
    print("-" * 40)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")
    
    # 6. Results Summary
    print("\n📈 STEP 6: RESULTS SUMMARY")
    print("-" * 40)
    
    print("Model Performance Comparison:")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name}: {accuracy:.4f}")
    
    # Best model detailed evaluation
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_tfidf)
    
    print(f"\nDetailed evaluation for best model ({best_model_name}):")
    print(classification_report(y_test, y_pred_best))
    
    # 7. Feature Analysis
    print("\n🔍 STEP 7: FEATURE ANALYSIS")
    print("-" * 40)
    
    if hasattr(best_model, 'coef_'):
        feature_importance = best_model.coef_[0] if len(best_model.coef_) == 1 else best_model.coef_[0]
        top_features_idx = np.argsort(np.abs(feature_importance))[-10:]
        
        print("Top 10 most important features:")
        for idx in reversed(top_features_idx):
            feature_name = feature_names[idx]
            importance = feature_importance[idx]
            print(f"  {feature_name}: {importance:.4f}")
    
    # 8. Sample Predictions
    print("\n🎯 STEP 8: SAMPLE PREDICTIONS")
    print("-" * 40)
    
    sample_texts = [
        "This Galaxy phone is amazing! Great camera and battery life.",
        "Terrible phone, constantly crashes and poor battery.",
        "It's okay, nothing special but works fine.",
    ]
    
    for text in sample_texts:
        cleaned = clean_text(text)
        vectorized = tfidf.transform([cleaned])
        prediction = best_model.predict(vectorized)[0]
        probability = max(best_model.predict_proba(vectorized)[0])
        print(f"  Text: '{text}'")
        print(f"  Prediction: {prediction} (confidence: {probability:.3f})")
        print()
    
    # 9. Project Structure Summary
    print("\n📁 STEP 9: PROJECT ARCHITECTURE OVERVIEW")
    print("-" * 40)
    
    print("Professional NLP project structure:")
    print("├── src/")
    print("│   ├── data_preprocessing.py    # Data loading & cleaning")
    print("│   ├── text_representation.py   # Feature extraction")
    print("│   ├── traditional_models.py    # ML models")
    print("│   ├── deep_learning_model.py   # LSTM implementation")
    print("│   └── utils.py                 # Utilities & setup")
    print("├── data/")
    print("│   └── sample_data.csv          # Android Galaxy reviews")
    print("├── notebooks/")
    print("│   └── Project_EMAI631.ipynb    # Research and development notebook")
    print("├── main.py                      # Main execution script")
    print("├── requirements.txt             # Dependencies")
    print("└── README.md                    # Documentation")
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Android Galaxy Sentiment Analysis project demonstrates")
    print("comprehensive NLP techniques with both traditional ML and")
    print("deep learning approaches. The modular architecture ensures")
    print("scalability and professional code organization.")
    print("=" * 60)

if __name__ == "__main__":
    main()
