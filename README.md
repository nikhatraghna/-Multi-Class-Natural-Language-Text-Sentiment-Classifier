# Sentiment Classifier — Twitter Airline Sentiment

Multi-class (negative/neutral/positive) text sentiment classifier using TF-IDF + Logistic Regression, trained on the [Kaggle Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

Runs as a single Google Colab notebook.

## Pipeline

Install deps → upload Kaggle API token → download dataset → preprocess text (clean, remove stop-words, lemmatize) → EDA (word clouds, n-grams, class balance) → TF-IDF vectorize → train Logistic Regression → evaluate (accuracy/precision/recall/F1, confusion matrix, learning curve) → test on new/unseen text.

## Setup

1. Open the notebook in [Colab](https://colab.research.google.com) and run cells top to bottom.
2. Get a Kaggle API token: kaggle.com → Settings → API → Create New Token → upload `kaggle.json` when prompted.

## Model

TF-IDF (unigrams+bigrams, 5000 features) → `LogisticRegression(class_weight="balanced")`

## Test on your own data

Last cell lets you upload any CSV with a `text` column (and optional `label` column) to get predictions or full evaluation metrics.

```csv
text,label
"Great flight, smooth boarding.",positive
"My luggage was lost.",negative
```
