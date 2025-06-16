# app.py - Flask Backend for MindScope
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded model components
model = None
label_encoder = None
sbert_model = None
stop_words = None

def load_model_components():
    """Load all necessary model components at startup"""
    global model, label_encoder, sbert_model, stop_words
    
    try:
        # Load your trained model (adjust path as needed)
        model = joblib.load('results\svm_model.pkl')  # or whatever your best model was
        label_encoder = joblib.load('results\label_encoder.pkl')
        
        # Load sentence transformer (make sure path matches your model)
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load stopwords
        stop_words = set(stopwords.words('english'))
        
        print("✅ All model components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False

def clean_text(text):
    """Clean text exactly like in your training code"""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def home():
    """Home endpoint to check if API is running"""
    return jsonify({
        "message": "🧠 MindScope API is running!",
        "status": "healthy",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided. Please send JSON with 'text' field."
            }), 400
        
        user_text = data['text'].strip()
        
        if not user_text:
            return jsonify({
                "error": "Empty text provided."
            }), 400
        
        # Clean the text
        cleaned_text = clean_text(user_text)
        
        if not cleaned_text:
            return jsonify({
                "error": "Text contains no valid words after cleaning."
            }), 400
        
        # Convert to embeddings using SBERT
        text_embedding = sbert_model.encode([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_embedding)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities if model supports it
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_embedding)[0]
            confidence = float(np.max(probabilities))
            
            # Create probability distribution
            prob_dict = {}
            for i, class_name in enumerate(label_encoder.classes_):
                prob_dict[class_name] = float(probabilities[i])
        else:
            confidence = 1.0  # If no probabilities available
            prob_dict = {predicted_class: 1.0}
        
        # Prepare response
        response = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "original_text": user_text,
            "cleaned_text": cleaned_text,
            "classes": label_encoder.classes_.tolist()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            "model_type": str(type(model).__name__),
            "classes": label_encoder.classes_.tolist(),
            "num_classes": len(label_encoder.classes_),
            "supports_probabilities": hasattr(model, 'predict_proba')
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "sbert_loaded": sbert_model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting MindScope API...")
    
    # Load model components at startup
    if load_model_components():
        print("🎯 Model loaded successfully!")
        print("📊 Available classes:", label_encoder.classes_.tolist())
        print("🔗 API endpoints:")
        print("   - GET  /          : Home page")
        print("   - POST /predict   : Make predictions")
        print("   - GET  /model-info: Model information")
        print("   - GET  /health    : Health check")
        print("\n🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model components. Please check your file paths.")