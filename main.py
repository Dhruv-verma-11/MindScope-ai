# 🧠 MindScope: Multiclass Mental Health Detection from Reddit Posts

# 📦 Step 1: Import libraries
import pandas as pd
import numpy as np
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs("results", exist_ok=True)


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 📂 Step 2: Load datasets
base_path = r"C:/Users/Dhruv/OneDrive/Desktop/MindScope/datasets"
train_df = pd.read_csv(f"{base_path}/posts_train.csv")
val_df = pd.read_csv(f"{base_path}/posts_val.csv")
test_df = pd.read_csv(f"{base_path}/posts_test.csv")

# 📊 Class distribution in validation set
val_df['class_name'].value_counts().plot(kind='bar', title='Class Distribution in Validation Set', color='skyblue', edgecolor='black')
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 🧹 Step 3: Clean post text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


print("Columns:", train_df.columns.tolist()) 

# 🧹 Step 3: Combine title and post, then clean
train_df['full_text'] = train_df['post']
val_df['full_text'] = val_df['post']
test_df['full_text'] = test_df['post']

train_df['clean_text'] = train_df['full_text'].apply(clean_text)
val_df['clean_text'] = val_df['full_text'].apply(clean_text)
test_df['clean_text'] = test_df['full_text'].apply(clean_text)


sbert_model = SentenceTransformer('./all-MiniLM-L6-v2')




# 🏷️ Step 4: Encode labels
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['class_name'])
val_df['label'] = le.transform(val_df['class_name'])
test_df['label'] = le.transform(test_df['class_name'])

# Show label mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# 🔠 Step 5: TF-IDF Vectorization
X_train = sbert_model.encode(train_df['clean_text'].tolist(), show_progress_bar=True)
X_val = sbert_model.encode(val_df['clean_text'].tolist(), show_progress_bar=True)
X_test = sbert_model.encode(test_df['clean_text'].tolist(), show_progress_bar=True)



y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

# 🤖 Step 6: Initialize Multiple Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB()
}


param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'Naive Bayes': {}  # No hyperparameters to tune for MultinomialNB
}


# 📊 Step 7: Train and Evaluate All Models
results = {}
predictions = {}

from sklearn.base import clone

print("🚀 Training and tuning models...\n")

for name, model in models.items():
    print(f"🔧 Tuning and training {name}...")
    if param_grids[name]:  # If we have parameters to tune
        grid = GridSearchCV(estimator=clone(model), param_grid=param_grids[name],
                            scoring='f1_weighted', cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"✅ Best parameters for {name}: {grid.best_params_}")
    else:
        best_model = clone(model)
        best_model.fit(X_train, y_train)
    
    # Predict
    y_pred = best_model.predict(X_val)
    predictions[name] = y_pred

    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    models[name] = best_model  # Replace old model with tuned one
    print(f"✅ {name} training done.\n")


print("\n" + "="*60)

# 📈 Step 8: Compare Model Performance
print("📊 MODEL COMPARISON RESULTS:")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df)

# 💾 Save model comparison results to CSV
results_df.to_csv("results/model_comparison_results.csv", index=True)
print("📁 Saved model comparison results to 'results/model_comparison_results.csv'")


# 📊 Visualize Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🧠 MindScope: Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = results_df[metric].values
    bars = ax.bar(results_df.index, values, color=colors[i], alpha=0.7, edgecolor='black')
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 📊 Step 9: Confusion Matrix Heatmaps for All Models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🧠 Confusion Matrix Comparison', fontsize=16, fontweight='bold')

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i//2, i%2]
    cm = confusion_matrix(y_val, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=le.classes_, yticklabels=le.classes_, 
                cmap="Blues", ax=ax)
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()

# 📋 Step 10: Detailed Classification Reports
print("\n" + "="*80)
print("📋 DETAILED CLASSIFICATION REPORTS:")
print("="*80)

for name, y_pred in predictions.items():
    print(f"\n🔍 {name.upper()}:")
    print("-" * 50)
    print(classification_report(y_val, y_pred, target_names=le.classes_))

# 🏆 Step 11: Find Best Model
best_model_name = results_df['F1-Score'].idxmax()
best_f1 = results_df.loc[best_model_name, 'F1-Score']

print(f"\n🏆 BEST PERFORMING MODEL:")
print(f"Model: {best_model_name}")
print(f"F1-Score: {best_f1:.4f}")

# Get the best model
best_model = models[best_model_name]

import joblib
# 💾 Save model, vectorizer, and label encoder
joblib.dump(best_model, f"results/{best_model_name.replace(' ', '_').lower()}_model.pkl")
#joblib.dump(vectorizer, "results/tfidf_vectorizer.pkl")
joblib.dump(le, "results/label_encoder.pkl")
print("💾 Saved best model and preprocessing components to 'results/'")


# 🧾 Display example predictions
sample_preds = pd.DataFrame({
    "Post": val_df['post'].head(10),
    "True Label": le.inverse_transform(y_val[:10]),
    "Predicted": le.inverse_transform(best_model.predict(X_val[:10]))
})
print("\n📌 SAMPLE PREDICTIONS:")
print(sample_preds[['Post', 'True Label', 'Predicted']])



# 🔢 Step 12: Probability Analysis with Best Model
print(f"\n🧠 Using {best_model_name} for probability analysis...")

if hasattr(best_model, 'predict_proba'):
    y_proba = best_model.predict_proba(X_val)
    
    # 📄 Convert to DataFrame for easy viewing
    proba_df = pd.DataFrame(y_proba, columns=le.classes_)
    proba_df['True Label'] = le.inverse_transform(y_val)
    proba_df['Predicted Label'] = le.inverse_transform(best_model.predict(X_val))
    
    # 🖨️ Show probabilities for the first 5 posts
    print("🔍 Example Post Probabilities (first 5 samples):")
    print(proba_df.head())
    
    # 📊 Show top class probabilities as percentage for a single post
    sample_index = 0
    print(f"\n🧠 Probability breakdown for Post #{sample_index}:")
    for class_name, prob in zip(le.classes_, y_proba[sample_index]):
        print(f"{class_name}: {prob:.2%}")

# 📊 Step 13: Enhanced Probability Visualization Function
def plot_probabilities_best_model(post_index):
    if hasattr(best_model, 'predict_proba'):
        probs = best_model.predict_proba(X_val)[post_index]
        classes = le.classes_
        
        plt.figure(figsize=(10, 6))
        import matplotlib.cm as cm
        import numpy as np

        cmap = cm.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(classes)))
        bars = plt.barh(classes, probs, color=colors)

        plt.xlabel("Probability")
        plt.title(f"🧠 {best_model_name}: Predicted Class Probabilities for Post #{post_index}")
        plt.xlim(0, 1)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"{best_model_name} doesn't support probability prediction.")

# 🧠 Example: Plot probabilities for post #0
plot_probabilities_best_model(0)

# ✍️ Step 14: Enhanced Custom Text Prediction Function
def predict_custom_text(text):
    clean = clean_text(text)
    vec = sbert_model.encode([clean])

    print(f"\n🔍 Prediction for: \"{text}\"")
    print("="*60)

    for name, model in models.items():
        pred_class = le.inverse_transform(model.predict(vec))[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            max_prob = np.max(proba)
            if max_prob >= 0.5:
                print(f"{name}: {pred_class} (confidence: {max_prob:.2%})")
            else:
                print(f"{name}: ❓ Not confident enough to predict (confidence: {max_prob:.2%})")
        else:
            print(f"{name}: {pred_class}")

    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(vec)[0]
        max_prob = np.max(proba)
        pred_class = le.classes_[np.argmax(proba)]

        print(f"\n🏆 Detailed analysis using {best_model_name}:")
        print("-" * 40)
        for cls, p in zip(le.classes_, proba):
            print(f"{cls}: {p:.2%}")

        if max_prob >= 0.5:
            print(f"\n🧠 Most likely condition: **{pred_class}** ({max_prob:.2%})")
        else:
            print(f"\n❗ Model is not confident enough to make a reliable prediction (max confidence: {max_prob:.2%})")

        plt.figure(figsize=(10, 6))
        bars = plt.barh(le.classes_, proba, color='green')
        plt.xlabel("Probability")
        plt.title(f"🧠 {best_model_name}: Predicted Class Probabilities")
        plt.xlim(0, 1)
        for bar, prob in zip(bars, proba):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{prob:.1%}', va='center', fontweight='bold')
        plt.tight_layout()
        plt.show()


# 🧪 Try with your own example:
while True:
    custom_text = input("\n📝 Enter a Reddit-style post to analyze (or type 'exit' to stop): ")
    if custom_text.strip().lower() == "exit":
        print("👋 Exiting interactive prediction mode.")
        break
    predict_custom_text(custom_text)


# 📊 Step 15: Model Performance Summary
print("\n" + "="*80)
print("📊 FINAL PERFORMANCE SUMMARY:")
print("="*80)
print(f"🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
print(f"📈 Total Models Compared: {len(models)}")
print(f"📋 Evaluation Metrics: Accuracy, Precision, Recall, F1-Score")
print(f"📊 Visualization: Confusion matrices and performance charts generated")
print("="*80)