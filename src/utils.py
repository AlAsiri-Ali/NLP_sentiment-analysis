import warnings
import nltk
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

def download_nltk_data():
    """Download required NLTK data"""
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
def setup_environment():
    """Setup the environment for the project"""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Download NLTK data
    download_nltk_data()
    
    print("Environment setup completed!")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def generate_classification_report(y_true, y_pred, target_names):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)