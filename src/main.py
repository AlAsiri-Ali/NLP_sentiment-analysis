"""
Sentiment Analysis of Android App Reviews
Main script to run the complete NLP pipeline
"""

import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_and_preprocess_data
from src.traditional_models import train_traditional_models
from src.deep_learning_model import train_lstm_model
from src.utils import setup_environment

def main():
    """Main function to run the sentiment analysis pipeline"""
    print("=== Sentiment Analysis of Android App Reviews ===")
    print("Setting up environment...")
    
    # Setup environment and download required data
    setup_environment()
    
    # Load and preprocess the dataset
    print("\n1. Loading and preprocessing data...")
    data_path = 'data/sample_data.csv'  # Using sample data for demonstration
    balanced_dataset = load_and_preprocess_data(data_path)
    
    # Train and evaluate traditional models
    print("\n2. Training traditional machine learning models...")
    train_traditional_models(balanced_dataset)
    
    # Train and evaluate deep learning model
    print("\n3. Training LSTM deep learning model...")
    model, history, le, tokenizer = train_lstm_model(balanced_dataset)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()