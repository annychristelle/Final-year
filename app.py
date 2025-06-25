from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
import os
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Flask Setup
app = Flask(__name__)
CORS(app)

app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "your-secret-key"),
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD")
)
mail = Mail(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load AI models
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
nlp_classifier = joblib.load('category_classifier.pkl')

# -------------------------
# TEXT PROCESSING FUNCTIONS
# -------------------------

def load_initial_data():
    logger.info("Checking and loading initial data from CSVs into Supabase if needed...")

    dataset_config = {
        "topics": "topics.csv",
        "anomalies": "anomalies.csv",
        "keywords": "keywords.csv",
        "feedback": "user_reviews.csv"
    }

    for table, csv_file in dataset_config.items():
        try:
            existing = supabase.table(table).select("*").limit(1).execute()
            if existing.data and len(existing.data) > 0:
                logger.info(f"✅ Table '{table}' already has data. Skipping CSV import.")
                continue

            if not os.path.exists(csv_file):
                logger.warning(f"⚠️ CSV file '{csv_file}' not found. Skipping...")
                continue

            df = pd.read_csv(csv_file)
            df.columns = [col.strip().lower() for col in df.columns]

            if table == "topics":
                df.columns = ["topic", "terms", "weight"]
                df["weight"] = df["weight"].astype(float)

            elif table == "anomalies":
                df.columns = ["batch", "negative_count", "z_score"]
                df["z_score"] = df["z_score"].astype(float)

            elif table == "keywords":
                df.columns = ["keyword", "score"]
                df["score"] = df["score"].astype(float)

            elif table == "feedback":
                df.columns = ["cleaned_review", "sentiment", "category", "review_length"]
                df["review_length"] = df["review_length"].astype(int)
                df['original_review'] = df['cleaned_review']

            records = df.to_dict(orient="records")
            if records:
                supabase.table(table).insert(records).execute()
                logger.info(f"✅ Inserted {len(records)} records into '{table}' from '{csv_file}'.")

        except Exception as e:
            logger.error(f"❌ Error processing table '{table}': {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    return ' '.join(tokens)

def classify_category(review):
    doc = nlp_classifier(review)
    return max(doc.cats, key=lambda x: doc.cats[x])

def send_alert(z_score, identifier, type='batch'):
    if z_score > 2:
        msg = Message('Feedback Alert', sender=os.getenv("MAIL_USERNAME"), recipients=['manager@example.com'])
        msg.body = f'{type.capitalize()} anomaly detected: {identifier} (Z-score: {z_score:.2f})'
        try:
            mail.send(msg)
            logger.info(f"Sent alert for {identifier}")
        except Exception as e:
            logger.error(f"Email failed: {e}")

# -------------------------
# ROUTES
# -------------------------

@app.route('/api/predict', methods=['POST'])
def predict_feedback():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Missing review text'}), 400

    review = data['review']
    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])
    sentiment = model.predict(X)[0]
    category = classify_category(cleaned)
    length = len(cleaned.split())

    return jsonify({
        "original_review": review,
        "cleaned_review": cleaned,
        "sentiment": sentiment,
        "category": category,
        "review_length": length
    }), 200

@app.route('/api/submit-review', methods=['POST'])
def submit_review():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Missing review field'}), 400

    review = data['review']
    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])
    sentiment = model.predict(X)[0]
    category = classify_category(cleaned)
    length = len(cleaned.split())

    try:
        supabase.table("feedback").insert({
            "original_review": review,
            "cleaned_review": cleaned,
            "sentiment": sentiment,
            "category": category,
            "review_length": length
        }).execute()
    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return jsonify({'error': 'Failed to save review'}), 500

    return jsonify({
        "original_review": review,
        "cleaned_review": cleaned,
        "sentiment": sentiment,
        "category": category,
        "review_length": length
    }), 200

@app.route('/api/insights', methods=['GET'])
def get_insights():
    load_initial_data()
    return jsonify({'message': 'Data loading triggered'}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

# -------------------------
# STARTUP
# -------------------------

def ensure_supabase_tables():
    logger.info("NOTE: Supabase table creation should be done manually or via SQL scripts.")

if __name__ == '__main__':
    ensure_supabase_tables()
    load_initial_data()
  app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
