# 🧠 MindScope – AI Mental Health Text Analyzer

**MindScope** is an AI-powered full-stack web application that analyzes Reddit-style user text and predicts potential mental health conditions using advanced NLP techniques and machine learning.

![MindScope Screenshot]("https://private-user-images.githubusercontent.com/175141672/455763363-b3451de4-914e-436b-9e0b-04756eb0b85b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTAxMjEwMzEsIm5iZiI6MTc1MDEyMDczMSwicGF0aCI6Ii8xNzUxNDE2NzIvNDU1NzYzMzYzLWIzNDUxZGU0LTkxNGUtNDM2Yi05ZTBiLTA0NzU2ZWIwYjg1Yi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNjE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDYxN1QwMDM4NTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMzdkOGNiMjA2ZGUxN2NiZmM1MmE3OTg2NTQ1MGViOWQyNzkxMzNhYzllNjEwOGM4ZWE0ZjdjZTEzMzU0M2VkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.LO3FIhhy0kF4k1LTw6BVocC6zTZ3khVo6BrxZxymGXU")


---

## 📌 Features

- 🔍 Predicts mental health conditions like **Depression**, **Anxiety**, **PTSD**, etc.
- 🧠 Leverages **SBERT (Sentence-BERT)** for powerful semantic embeddings
- 🤖 Compares and trains multiple ML models (SVM, Random Forest, Naive Bayes)
- 📊 Interactive prediction confidence chart (via **Chart.js**)
- 🌐 REST API powered by **Flask**
- 🔄 End-to-end system: from training → model saving → live prediction

---

## 🧰 Tech Stack

| Frontend                  | Backend             | ML / NLP                        |
|---------------------------|---------------------|----------------------------------|
| HTML, CSS, JavaScript, Chart.js | Python, Flask, Flask-CORS | SBERT, scikit-learn, NLTK, joblib |

---

## 🎯 ML Workflow

1. Clean + preprocess mental health posts from Reddit
2. Encode text with **SBERT** sentence embeddings
3. Train multiple models and evaluate using **F1-score**
4. Select and save the best-performing model as `.pkl`
5. Serve the model via Flask for real-time predictions

---

## 🖥️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruv-verma-11/MindScope-ai.git
cd MindScope-ai

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Run the flask app

```bash
pyhton app.py


### 4.📁 Project Structure

```bash
MindScope-ai/
├── app.py              # Flask backend
├── main.py             # Model training + saving
├── results/            # Saved model + label encoder
├── templates/          # index.html (UI)
├── static/             # Chart.js and styling
├── requirements.txt    # Python dependencies
├── render.yaml         # (Optional) Render deployment config
└── README.md

### 5.⚠️ Disclaimer

```bash
This project is for educational and demonstration purposes only.
It is not intended to diagnose or treat any mental health condition.
Please consult a qualified professional for support.

### 6. How to use-

```bash

git add README.md
git commit -m "Add polished README"
git push

---


### ✅ 7. Save and push to GitHub:

```bash
git add README.md
git commit -m "Add detailed project README"
git push origin main
