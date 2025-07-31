# EmotionSense – Real-Time Emotion Detection from Text using NLP & Machine Learning

**EmotionSense** is a full-stack web application that detects and classifies human emotions from user-inputted text in real time. Using Natural Language Processing (NLP) and machine learning, the app maps words to emotions like **joy**, **anger**, **fear**, and **sadness**—providing an intuitive way to understand emotional states through text.

---

## Use Case: Supporting Mental Health Awareness

EmotionSense can be used as a mental health support tool to help individuals:
- Reflect on their emotional states through journaling or chat logs.
- Identify patterns of emotions over time.
- Support therapists in sentiment tracking.
- Build emotion-aware chatbots or educational tools.

---

## Tech Stack

### Frontend
- React.js (with functional components)
- Axios (for API requests)
- Bootstrap (for aesthetic, responsive design)

### Backend
- Python 3
- Flask (for RESTful API)
- scikit-learn, pandas, numpy
- NLTK & spaCy (for NLP tasks)
- Joblib (for model serialization)

---

## Dataset

- **File**: `train_emotions.csv`
- **Source**: [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- **Columns**:
  - `content`: text input (e.g., "I feel amazing today!")
  - `label`: emotion class (e.g., joy, anger, fear, etc.)

---

## Project Pipeline

### 1. **Data Preprocessing**
- Lowercasing, punctuation & number removal
- Tokenization with `nltk`
- Stopword removal
- Lemmatization with `spaCy`

### 2. **Feature Engineering**
- TF-IDF Vectorization (top 5000 terms)

### 3. **Model Training**
- Logistic Regression (`liblinear` solver)
- Multi-class classification

### 4. **API Development**
- Flask API with `/api/analyze` POST route
- Returns predicted emotion for a given text

### 5. **Frontend Integration**
- React app sends text input to backend
- Displays predicted emotion with smooth UI

---

## Example Predictions

```python
predict_emotion("I'm feeling on top of the world!")      # ➤ joy  
predict_emotion("Why does everything feel so heavy?")     # ➤ sadness  
predict_emotion("That scared me a lot!")                  # ➤ fear 
