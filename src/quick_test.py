"""
Quick test script to verify the sentiment analysis pipeline works
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✓ NLTK data downloaded successfully")
except:
    print("⚠ NLTK download may have failed, but continuing...")

def basic_preprocessing(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def main():
    print("=== Quick Sentiment Analysis Test ===")
    
    # Load data
    print("1. Loading sample data...")
    try:
        df = pd.read_csv('data/sample_data.csv')
        print(f"   Data loaded successfully: {df.shape[0]} samples")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Label distribution:")
        print(df['label'].value_counts())
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Basic preprocessing
    print("\n2. Preprocessing text...")
    try:
        df['processed_text'] = df['comment_text'].apply(basic_preprocessing)
        print(f"   Text preprocessing completed")
        print(f"   Sample processed text: {df['processed_text'].iloc[0][:100]}...")
    except Exception as e:
        print(f"   Error in preprocessing: {e}")
        return
    
    # Split data
    print("\n3. Splitting data...")
    try:
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
    except Exception as e:
        print(f"   Error splitting data: {e}")
        return
    
    # Vectorization
    print("\n4. Text vectorization...")
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print(f"   Vectorization completed: {X_train_vec.shape}")
    except Exception as e:
        print(f"   Error in vectorization: {e}")
        return
    
    # Train model
    print("\n5. Training logistic regression model...")
    try:
        model = LogisticRegression(random_state=42)
        model.fit(X_train_vec, y_train)
        print("   Model training completed")
    except Exception as e:
        print(f"   Error training model: {e}")
        return
    
    # Evaluate
    print("\n6. Evaluating model...")
    try:
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Classification Report:")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"   Error in evaluation: {e}")
        return
    
    print("\n=== Quick Test Complete ===")
    print("✓ Basic sentiment analysis pipeline is working!")

if __name__ == "__main__":
    main()
