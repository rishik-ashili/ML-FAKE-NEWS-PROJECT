# Fake News Detection Using Machine Learning

This repository contains the implementation of a robust machine learning pipeline for detecting fake news articles. The project leverages state-of-the-art natural language processing (NLP) techniques, particularly the Multilingual BERT model, to classify news articles as either fake or real.

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Challenges](#challenges)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Technical Details](#technical-details)
- [Performance](#performance)
- [Team Members](#team-members)
- [GitHub Repository Link](#github-repository-link)

## Introduction

With the surge of digital content, fake news has become a critical concern. This project aims to address the problem by developing a machine learning model that classifies news articles into fake or real categories with high accuracy. 

## Motivation

- **Why This Project?**
  - Fake news has widespread societal and political implications.
  - It is crucial to develop automated tools for early detection to combat misinformation.

## Challenges

1. **Data Quality:** 
   - Handling noisy, unstructured multilingual text data.
2. **Class Imbalance:**
   - Tackling the skewed distribution of real versus fake news data.
3. **Feature Extraction:**
   - Efficiently extracting meaningful features for improved classification.

## Key Features

- **Multilingual Compatibility:** Supports text in multiple languages (e.g., English, Hindi) through preprocessing and tokenization.
- **Advanced NLP Techniques:** Leveraging pre-trained Multilingual BERT for feature extraction.
- **Explainability:** Identifies key features contributing to classification.

## Methodology

### Pipeline Overview
1. **Data Preprocessing:**
   - Tokenization, stemming, and lemmatization.
   - Text cleaning to remove noise and irrelevant elements.
2. **Feature Engineering:**
   - Utilized **BERT Tokenizer** for transforming text into embeddings.
3. **Model Training:**
   - Fine-tuned Multilingual BERT with additional dense layers for classification.
   - Training performed on Kaggle using NVIDIA GPU for 10+ hours.
4. **Evaluation:**
   - Model performance evaluated using metrics like accuracy, precision, recall, and F1-score.

### Model Architecture
- **Base Model:** Pre-trained `bert-base-multilingual-cased` from Hugging Face Transformers.
- **Custom Layers:**
  - Fully connected dense layers with ReLU activations.
  - Softmax activation in the output layer for binary classification.
- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Adam with a learning rate of `1e-5`.

## Technical Details

- **Programming Language:** Python
- **Major Libraries:**
  - `torch` and `torch.nn` for deep learning.
  - `transformers` for pre-trained BERT models.
  - `scikit-learn` for data splitting and metrics.
- **Dataset:**
  - A combination of English and Hindi news articles with labeled fake and real news.

## Performance

The final model achieved a **validation accuracy of 89.86%**. Below are detailed performance metrics:

- **Accuracy:** 89.86%
- **Precision:** Balanced between fake and real labels.
- **Recall:** High sensitivity for detecting fake news.
- **F1-Score:** Balanced metric showing the model's reliability.

### Key Observations
- **High Accuracy:** The model generalizes well to unseen data.
- **Multilingual Capability:** Processes both Hindi and English datasets efficiently.

## GitHub Repository Link

[Project Repository](https://github.com/rishik-ashili/ML-FAKE-NEWS-PROJECT/tree/master)   

## Kaggle Notebook Link (Model training)

[Kaggle Notebook](https://www.kaggle.com/code/rishikashili/ml-fake-news)   

## Resources used (csv files and model pkl)  

[Resources GDrive](https://drive.google.com/drive/folders/12UotjF9RAj4yCq-TRHFySdT74NDJryJu?usp=sharing)   

