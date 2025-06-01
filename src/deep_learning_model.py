import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_lstm_model(balanced_dataset):
    """Train and evaluate LSTM model for sentiment analysis"""
    # Extract cleaned text and labels
    texts = balanced_dataset['cleand_text'].values
    labels = balanced_dataset['label'].values

    # Encode labels into integers, then convert to one-hot vectors
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    y = to_categorical(labels_encoded)

    # Split the dataset into training and testing sets (stratified)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, stratify=y, test_size=0.3, random_state=42
    )

    # Set tokenizer and padding parameters
    max_words = 10000   # Max number of words to keep (vocabulary size)
    max_len = 100       # Max length of each sequence (number of tokens)

    # Initialize tokenizer and fit it on the training text
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    # Pad sequences to ensure consistent input length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))  # Word embeddings
    model.add(Bidirectional(LSTM(64)))  # Bidirectional LSTM layer with 64 units
    model.add(Dropout(0.5))             # Dropout to reduce overfitting
    model.add(Dense(64, activation='relu'))  # Dense layer with ReLU activation
    model.add(Dense(2, activation='softmax'))  # Output layer for 2 classes

    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Further split training data into train and validation sets
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(
        X_train_pad, y_train, test_size=0.2, random_state=42
    )

    # Define early stopping to stop training when validation loss doesn't improve
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=128,
        callbacks=[early_stop]
    )

    # Predict class probabilities on the test set
    y_pred_prob = model.predict(X_test_pad)

    # Convert probabilities to class predictions (using the index of the highest probability)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Convert one-hot encoded true labels back to class indices
    y_true = np.argmax(y_test, axis=1)

    # Includes precision, recall, F1-score, and support for each class
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)

    # Print classification report in percentage format
    print("\n=== LSTM CLASSIFICATION REPORT (%) ===")
    print(f"{'Class':<12}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Support':>12}")
    print("-" * 60)

    for label in le.classes_:
        precision = report[label]['precision'] * 100
        recall = report[label]['recall'] * 100
        f1 = report[label]['f1-score'] * 100
        support = int(report[label]['support'])
        print(f"{label:<12}{precision:12.2f}{recall:12.2f}{f1:12.2f}{support:12}")

    # Print overall accuracy of the model
    accuracy = report['accuracy'] * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Confusion matrix visualization
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - LSTM')
    plt.show()

    # Display model architecture summary
    model.summary()
    
    return model, history, le, tokenizer

def create_lstm_model(max_words, max_len):
    """Create LSTM model architecture"""
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def prepare_data(texts, labels, max_words=10000, max_len=100):
    """Prepare data for LSTM training"""
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    y = to_categorical(labels_encoded)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, stratify=y, test_size=0.3, random_state=42
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, le

def train_model(model, X_train, y_train, X_valid, y_valid):
    """Train the model with early stopping"""
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=128,
        callbacks=[early_stop]
    )
    return history

def predict(model, X_test):
    """Make predictions using the trained model"""
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    return y_pred