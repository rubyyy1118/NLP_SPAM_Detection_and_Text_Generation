# Natural Language Processing for Spam Detection and Text Generation
## 1. Project Overview

This repository applies traditional machine learning (SVM) and deep learning (LSTM) techniques to classify SMS messages as **spam** or **ham**. The project also includes a text generation model that produces spam-like messages, which are subsequently used to evaluate the performance of the classifiers.

### Key Objectives:
- **Spam Detection**: Using different feature extraction techniques (TF-IDF and Word2Vec) to classify messages into **spam** or **ham**.
- **Text Generation**: Implementing an LSTM-based model to generate spam-like messages and test their classification.

## 2. Files in this Repository

- `Natural Language Analysis Report.pdf`: The report contains detailed explanations and results for both SVM and LSTM models, along with the text generation task.
- `NLP_code.ipynb`: The Jupyter notebook containing the code used to preprocess the data, train the models, and generate spam text.
- `SpamVsHam.tsv`: The original dataset consists of SMS messages labeled spam or ham.

## 3. Instructions for Running the Code

To run the notebook (`NLP_code.ipynb`), ensure you have the following installed:
- **Python 3.13.0**
- Libraries: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `keras`, `nltk`

### Steps:
1. Install the necessary libraries using:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn tensorflow keras nltk
   ```
2. Open the Jupyter notebook in your environment and run the cells sequentially to replicate the results.
3. The notebook includes:
   - Data preprocessing (handling missing values, tokenization, and vectorization).
   - SVM model training with different feature extraction techniques (TF-IDF and Word2Vec).
   - LSTM model training using both TF-IDF and Word2Vec features.
   - Text generation using a Recurrent Neural Network (LSTM) to generate spam-like messages.
   - Testing the classifiers using both real and generated data.

### Expected Outputs:
- **Performance Metrics**: Accuracy, precision, recall, and F1-scores for both SVM and LSTM models.
- **Text Generation**: Generated spam-like messages used to evaluate the classifiers.

## 4. Analysis Overview

### 4.1 Feature Extraction Techniques
- **TF-IDF**: Captures the importance of words based on their frequency across the corpus.
- **Word2Vec**: Creates dense word embeddings that capture semantic relationships between words.

### 4.2 Handling Data Imbalance
- The dataset is imbalanced, with **86.6%** of the messages labeled as ham and only **13.4%** as spam. This imbalance can lead to biased models that perform well for ham messages but poorly for spam.
- **Class Weighting**: To address this, class weights were applied during model training. This technique assigns higher weights to the minority class (spam) to encourage the model to pay more attention to it.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: I also considered an alternative to balance the classes by generating synthetic spam samples, but the best results were achieved using class weighting.

### 4.3 SVM Model Results
- The SVM model was trained on both TF-IDF and Word2Vec features.
- **Best results**: The SVM with TF-IDF features achieved an accuracy of **98%**, significantly outperforming the Word2Vec-based model, especially in identifying spam messages.
  - With **class weighting**, the recall for spam improved, reducing the number of missed spam messages.

### 4.4 LSTM Model Results
- The LSTM model was trained using both TF-IDF and Word2Vec features.
- **Best results**: The LSTM model with TF-IDF achieved **98% accuracy**, but training with Word2Vec required more computational resources and slightly decreased accuracy.
  - Class weights were also applied to the LSTM model, improving the model’s ability to identify spam messages, though at the cost of slightly increased false positives for ham messages.

### 4.5 Text Generation Results
- The text generation model produced spam-like messages using the LSTM architecture.
- These generated messages were used to evaluate the SVM and LSTM classifiers. After applying class weights, the SVM model correctly classified **95%** of the generated spam, improving performance in handling spam detection.

## 5. Conclusion

This project successfully demonstrates the application of traditional machine learning (SVM) and deep learning (LSTM) techniques for spam detection. Addressing the data imbalance through class weighting significantly improved the recall for spam messages. Additionally, the text generation model highlights the ability of neural networks to generate coherent spam-like content, which can be used to evaluate classifier performance.

## 6. License

This project is for academic purposes only as part of the Natural Language Analysis module.
