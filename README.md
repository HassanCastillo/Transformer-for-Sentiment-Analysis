# Transformer-for-Sentiment-Analysis
A transformer-based sentiment analysis model trained on large-scale IMDB reviews. This project captures nuanced language patterns to accurately classify user sentiment (positive/negative), enabling deeper insights for personalization systems and improving overall user engagement.

Project Features:

✔ Custom Transformer Architecture:
- Manual implementation of
- AttentionHead
- MultiHeadAttention
- FeedForward network
- Transformer Block
- Positional + Token embeddings
- Classification head
- Built using PyTorch

✔ Subword Tokenization:
Uses BERT’s bert-base-uncased subword tokenizer
Efficient vocabulary size
Better handling of rare/unseen words

✔ Full Training Pipeline:
Data loading from IMDB dataset
Custom IMDBDataset class
PyTorch DataLoaders for train/val/test
Training loop with AdamW optimizer
Validation + test accuracy computation

✔ Sentiment Analysis:
Binary classification (Positive / Negative)
Mean pooling for sequence representation
CrossEntropy loss for training

Results Summary:

This project builds a sentiment-classification pipeline using a GPT-style transformer trained from scratch. A custom tokenizer, dataset loader, and DemoGPT model were implemented to classify positive and negative text reviews. The model was trained for four epochs with AdamW and cross-entropy loss, showing steady decreases in training loss and improvements in validation accuracy. Final evaluation on the test set using a custom calculate_accuracy() function yielded 77.56% accuracy, demonstrating a working end-to-end transformer approach for sentiment analysis.

Key Takeaways:

- A Transformer can be built from scratch using PyTorch without relying on pre-trained models.
- Using a subword tokenizer such as bert-base-uncased improves vocabulary efficiency and helps the model generalize better to rare or unseen words, often resulting in higher accuracy.
- Tokenization and padding shape the input that Transformers expect.
- Evaluation pipelines (validation & test loaders) ensure model generalization.
- Dataset integrity checks help catch issues early: correct tensor shapes, valid labels and non-empty datasets.
- Transformers can perform competitively even with a small architecture and limited training epochs.
- Training stability improves with: AdamW optimizer, proper learning rate, normalized embeddings, and attention masking.

Technologies Used:

Python,
PyTorch,
HuggingFace Tokenizers,
NumPy / Pandas,
Matplotlib,
IMDB Movie Reviews Dataset
