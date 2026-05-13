# Toxic Comment Detection using DistilBERT

## Overview

This project focuses on detecting toxic comments using Natural Language Processing (NLP) and a Transformer-based deep learning model.

The system classifies comments into multiple toxicity categories such as toxic, obscene, insult, threat, and identity hate. Since a single comment can belong to more than one category, the task is treated as a multi-label classification problem.

The project began with a traditional machine learning pipeline using TF-IDF and Logistic Regression. After evaluating its limitations in understanding context and semantic meaning, the approach was upgraded to DistilBERT for better performance.

---

## Categories Predicted

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* HuggingFace Transformers
* DistilBERT

---

## Dataset

The project uses the Jigsaw Toxic Comment Classification dataset.

The dataset contains user comments along with binary labels representing different forms of toxicity.

---

## Approach

### 1. Data Preparation

* Loaded and explored the dataset
* Split data into training and validation sets
* Prepared labels for multi-label classification

### 2. Baseline Model

A traditional ML baseline was created using:

* TF-IDF Vectorization
* Logistic Regression

Although the baseline produced reasonable results, it struggled with contextual understanding.

### 3. Transformer Model

The final implementation uses DistilBERT:

* Pretrained transformer model
* Faster and lighter than BERT
* Better contextual understanding for NLP tasks

Tokenization was performed using the DistilBERT tokenizer, converting comments into input IDs and attention masks.

### 4. Training

The model was trained using the HuggingFace Trainer API.

Key settings:

* Learning rate: 2e-5
* Weight decay: 0.01
* Epochs: 1

A reduced subset of the dataset was used during experimentation to reduce training time on local hardware.

---

## Evaluation

The model was evaluated using F1-score because the dataset is imbalanced.

### Results

| Metric         | Score |
| -------------- | ----- |
| Micro F1 Score | 0.75  |
| Macro F1 Score | 0.40  |

The Micro F1 score indicates good overall performance, while the lower Macro F1 score reflects difficulty in predicting less frequent classes such as threat and identity hate.

---

## Sample Prediction

```python
predict_toxicity("You are stupid and I hate you")
```

Example output:

```text
Toxic: 1
Insult: 1
Threat: 0
Identity Hate: 0
```

---

## Project Structure

```text
toxic-comment-detection/
│
├── toxic_model/
├── notebook.ipynb
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Challenges Faced

* Dependency conflicts during setup
* GPU configuration issues
* Multi-label evaluation handling
* Long training time on CPU

---

## Future Improvements

* Train on the complete dataset using GPU
* Improve performance on minority classes
* Build a Streamlit-based interface
* Deploy the model for real-time moderation

---

## Conclusion

This project demonstrates the use of Transformer-based NLP models for toxic comment classification. The shift from traditional machine learning to DistilBERT improved the model’s ability to understand context and handle multi-label predictions more effectively.

---

## Author

Anshuman 
