import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
import nltk

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

def load_dataset(file_path):
    """Load the CSV file"""
    dataset = pd.read_csv(file_path)
    return dataset

def explore_dataset(dataset):
    """Explore the dataset"""
    print("Dataset shape and label distribution:")
    print(dataset.shape, dataset['label'].value_counts())
    print("\nFirst 10 rows:")
    print(dataset.head(10))
    print("\nDataset info:")
    print(dataset.info())
    print("\nDataset description:")
    print(dataset.describe())
    print("\nNull values:")
    print(dataset.isnull().sum())

def remove_neutral_labels(dataset):
    """Remove neutral labels from dataset"""
    dataset = dataset[dataset['label'] != 'neutral']
    print("After removing neutral labels:")
    print(dataset['label'].value_counts())
    return dataset

def balance_dataset(dataset):
    """Balance the dataset by upsampling positive class"""
    # Separate the dataset into positive and negative classes
    df_negative = dataset[dataset['label'] == 'negative']
    df_positive = dataset[dataset['label'] == 'positive']

    # Upsample the positive class to match the size of the negative class
    df_positive_upsampled = resample(
        df_positive,
        replace=True,                 # Sample with replacement
        n_samples=len(df_negative),   # Match number of samples in the negative class
        random_state=42               # Ensure reproducibility
    )

    # Combine the negative class with the upsampled positive class
    balanced_dataset = pd.concat([df_negative, df_positive_upsampled])

    # Shuffle the dataset to mix positive and negative samples
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print the distribution of labels to confirm balance
    print(balanced_dataset['label'].value_counts())
    
    return balanced_dataset

def preprocess_text(text):
    """Preprocess text data"""
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Combine words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

def apply_text_preprocessing(dataset):
    """Apply the preprocessing function to each comment in the 'comment_text' column"""
    # This includes cleaning, tokenization, stopword removal, and lemmatization
    dataset['cleand_text'] = dataset['comment_text'].apply(preprocess_text)

    # Display the first 5 rows of the dataset to inspect the cleaned text
    print("Processed dataset:")
    print(dataset.head())
    print("\nDataset columns:")
    print(dataset.columns)
    
    return dataset

def load_and_preprocess_data(file_path):
    """Complete data loading and preprocessing pipeline"""
    # Load the dataset
    dataset = load_dataset(file_path)
    
    # Explore the dataset
    explore_dataset(dataset)
    
    # Remove neutral labels
    dataset = remove_neutral_labels(dataset)
    
    # Balance the dataset
    balanced_dataset = balance_dataset(dataset)
    
    # Apply text preprocessing
    balanced_dataset = apply_text_preprocessing(balanced_dataset)

    return balanced_dataset