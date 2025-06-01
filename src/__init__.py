"""
Sentiment Analysis Package
Author: NLP Project Team
Description: Sentiment analysis of Android Galaxy app reviews using traditional ML and deep learning approaches
"""

__version__ = "1.0.0"
__author__ = "NLP Project Team"

# Import main functions for easy access
from .data_preprocessing import load_and_preprocess_data, preprocess_text
from .text_representation import text_representations
from .traditional_models import train_traditional_models
from .deep_learning_model import train_lstm_model
from .utils import setup_environment

__all__ = [
    "load_and_preprocess_data",
    "preprocess_text", 
    "text_representations",
    "train_traditional_models",
    "train_lstm_model",
    "setup_environment"
]