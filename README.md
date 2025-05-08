# Multimodal Emotion Classification from Speech using CNNs and RNNs

## Objective

The goal of this project is to design and train a deep learning model capable of classifying human emotions using two modalities derived from speech:
- **Audio data** (converted into spectrograms or MFCCs for CNN input)
- **Textual transcripts** (generated from speech via speech-to-text for NLP modeling)

This task explores the use of Convolutional Neural Networks (CNNs) for image-like data and Recurrent Neural Networks (RNNs) for sequential text data, with the potential for multimodal learning in future phases.

---

## Dataset

**RAVDESS Emotional Speech Audio**  
- Source: [Kaggle - RAVDESS Audio Only](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- 1440 labeled speech clips (both male and female actors)
- 8 emotional classes:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

---

## Phase 1 – Unimodal Pipelines

### 1. Audio CNN Pipeline

- **Preprocessing**:
  - Converted audio clips into Mel spectrograms
  - Normalized and resized for CNN input

- **Model Architecture**:
  - Two configurations tested:
    - CNN with 2 convolutional layers
    - CNN with 3 convolutional layers

- **Results**:
  - Validation Accuracy (2 conv layers): **41%**
  - Validation Accuracy (3 conv layers): **38%**

---

### 2. Text RNN Pipeline

- **Preprocessing**:
  - Speech-to-text conversion using OpenAI Whisper
  - Cleaned and tokenized text data
  - Padded sequences for input to RNN

- **Model Architecture**:
  - GRU-based RNN for emotion classification from text

- **Results**:
  - Validation Accuracy: **10%**

---

## Observations and Analysis

### CNN Performance

- **Strengths**:
  - Able to learn from spectrogram patterns associated with vocal tone, pitch, and energy
- **Weaknesses**:
  - Overfitting observed due to small dataset size
  - Adding more layers did not improve performance, likely due to insufficient data and over-complexity

### RNN Performance

- **Major Issues**:
  - Very low accuracy due to poor-quality transcripts(most transcripts were the sentences as it is
    and since there were only two sentences there is not much to infer from transcripts. Few had an ! mark, rest most were just assertive sentences.)
  - i.e. most transcripts were garbled or missing emotional context entirely

---

## Reasons for Poor Performance

### Common Issues:
- **Small dataset size**: Deep models require more data to generalize well
- **Class imbalance**: Some emotions are underrepresented
- **No data augmentation**: Limited exposure to variability
- **Low transcript quality**: Especially harmful for text-based models
- **Emotion ambiguity**: Emotions in speech may overlap or be subtle

---

## Suggestions for Improvement

### Audio-CNN:
- Use **data augmentation** techniques (e.g., pitch shift, noise addition)

### Text-RNN:
- Fine-tune or use **better STT models**

### General:
- Combine audio and text in a **multimodal model** (Phase 2)
- Perform **hyperparameter tuning** for both CNN and RNN
- Use **attention mechanisms** to focus on emotion-relevant parts of audio/text

---

## Future Work

- **Phase 2**: Develop a multimodal architecture combining both spectrogram-based CNN and transcript-based RNN/Transformer branches.
- Evaluate fusion strategies: early, late, or hybrid fusion.
- Experiment with Transformer-based architectures for both audio and text (e.g., AST, BERT).
Find it here: https://github.com/akshat-31415/emotion_classifier_updated
---
