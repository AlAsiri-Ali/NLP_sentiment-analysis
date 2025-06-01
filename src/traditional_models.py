from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
try:
    from .text_representation import text_representations
except ImportError:
    from text_representation import text_representations

def train_traditional_models(balanced_dataset):
    """Train and evaluate traditional ML models with different text representations"""
    # Define text representation techniques to evaluate
    representations = ['bow', 'tfidf', 'word2vec']

    # Define traditional ML classification models with class weight balancing
    models = {
        'Logistic Regression': LogisticRegression(solver='lbfgs', class_weight='balanced'),
        'Support Vector Classifier': SVC(class_weight='balanced'),
        'Random Forest Classifier': RandomForestClassifier(class_weight='balanced')
    }

    # Encode labels (e.g., 'positive', 'negative') into numerical format
    le = LabelEncoder()
    y = le.fit_transform(balanced_dataset['label'])

    # Split the dataset into training and testing sets (stratified to maintain class balance)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        balanced_dataset['cleand_text'], y, stratify=y, test_size=0.3, random_state=42
    )

    # Loop through each model and each text representation technique
    for algo_name, model in models.items():
        for rep in representations:
            print(f"\n=== {algo_name} with {rep.upper()} ===")

            # Apply the chosen text representation method
            X_train, X_test, _ = text_representations(X_train_text, X_test_text, rep)

            # Train the model on the training data
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Print evaluation metrics
            print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
            print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
            print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
            print(f"F1-Score:  {f1_score(y_test, y_pred) * 100:.2f}%")

            # Generate and display the confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {algo_name} with {rep}')
            plt.show()

def train_and_evaluate_models(X, y):
    """Original function for training and evaluating models"""
    models = {
        'Logistic Regression': LogisticRegression(solver='lbfgs', class_weight='balanced'),
        'Support Vector Classifier': SVC(class_weight='balanced'),
        'Random Forest Classifier': RandomForestClassifier(class_weight='balanced')
    }
    
    results = {}
    
    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    return results

def encode_labels(y):
    """Encode labels to numerical format"""
    le = LabelEncoder()
    return le.fit_transform(y)