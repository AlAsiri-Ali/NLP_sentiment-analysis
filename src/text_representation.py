import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def text_representations(X_train_text, X_test_text, representation_type, vector_size=100):
    """Apply different text representation techniques"""
    # Bag of Words representation
    if representation_type == 'bow':
        vectorizer = CountVectorizer()
        # Fit the vectorizer on training data and transform both train and test sets
        X_train_transformed = vectorizer.fit_transform(X_train_text)
        X_test_transformed = vectorizer.transform(X_test_text)
        return X_train_transformed, X_test_transformed, vectorizer

    # TF-IDF representation
    elif representation_type == 'tfidf':
        vectorizer = TfidfVectorizer()
        # Fit the vectorizer on training data and transform both train and test sets
        X_train_transformed = vectorizer.fit_transform(X_train_text)
        X_test_transformed = vectorizer.transform(X_test_text)
        return X_train_transformed, X_test_transformed, vectorizer

    # Word2Vec representation
    elif representation_type == 'word2vec':
        # Tokenize each sentence (split by whitespace)
        train_tokenized = [sentence.split() for sentence in X_train_text]
        test_tokenized = [sentence.split() for sentence in X_test_text]

        # Train Word2Vec model on training data
        model = Word2Vec(train_tokenized, vector_size=vector_size, window=5, min_count=1, workers=4)

        # Average word vectors for each sentence in training data
        X_train_transformed = np.array([
            np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
            if any(word in model.wv for word in sentence) else np.zeros(vector_size)
            for sentence in train_tokenized
        ])

        # Average word vectors for each sentence in test data
        X_test_transformed = np.array([
            np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
            if any(word in model.wv for word in sentence) else np.zeros(vector_size)
            for sentence in test_tokenized
        ])

        return X_train_transformed, X_test_transformed, model

    # Raise an error for invalid representation types
    else:
        raise ValueError("Invalid representation type. Choose 'bow', 'tfidf', or 'word2vec'.")