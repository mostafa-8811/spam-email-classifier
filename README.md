# Spam Email Classifier using NLP & Logistic Regression

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline that preprocesses raw email text, extracts features using TF-IDF vectorization, and classifies messages as Spam or Ham (Not Spam).

## Features
- **NLP Preprocessing:** Text cleaning using regular expressions, NLTK stop-word removal, and Porter Stemming.
- **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with a 3,000-feature vocabulary limit.
- **Model Serialization:** Automatically exports trained model and vectorizer binaries (`.pkl`) via `joblib` for rapid inference without retraining.
- **Single-Text Inference Engine:** Modular pipeline capable of processing and classifying individual raw strings.

## Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NLTK, Scikit-Learn, Joblib

## Performance Metrics
The Logistic Regression model was evaluated on a test dataset split, achieving **95.52% overall accuracy**:

```text
Model Accuracy: 95.52%

Classification Report:
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       965
           1       0.96      0.69      0.81       150

    accuracy                           0.96      1115
   macro avg       0.96      0.84      0.89      1115
weighted avg       0.96      0.96      0.95      1115
