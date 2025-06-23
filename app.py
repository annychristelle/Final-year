from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
from flask_mail import Mail, Message
import os
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='your-email@gmail.com',  # Replace with your Gmail
    MAIL_PASSWORD='your-app-password'      # Replace with your App Password
)
mail = Mail(app)

# Load models and data
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    nlp_classifier = joblib.load('category_classifier.pkl')
    topics_df = pd.read_csv('topics.csv')
    anomalies_df = pd.read_csv('anomalies.csv')
    keywords_df = pd.read_csv('keywords.csv')
    categorized_df = pd.read_csv('categorized_reviews.csv')
    logger.info("Loaded models and static data files")
except FileNotFoundError as e:
    logger.error(f"Required file not found: {e}. Run train_model.py and related scripts first.")
    exit(1)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def classify_category(review):
    doc = nlp_classifier(review)
    return max(doc.cats, key=lambda x: doc.cats[x])

def send_alert(z_score, identifier, type='batch'):
    if z_score > 2:
        msg = Message('Feedback Alert', sender='your-email@gmail.com', recipients=['manager@example.com'])
        msg.body = f'{type.capitalize()} anomaly detected in {type} {identifier} with Z-score {z_score:.2f}'
        try:
            mail.send(msg)
            logger.info(f"Alert sent for {type} {identifier}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    category = None
    user_reviews_file = 'user_reviews.csv'
    if os.path.exists(user_reviews_file):
        try:
            user_reviews_df = pd.read_csv(user_reviews_file)
            logger.info(f"Loaded {len(user_reviews_df)} user reviews")
        except Exception as e:
            logger.error(f"Error loading {user_reviews_file}: {e}")
            user_reviews_df = pd.DataFrame(columns=['Cleaned_Review', 'Sentiment', 'Category', 'Review_Length'])
    else:
        logger.warning("No user_reviews.csv found")
        user_reviews_df = pd.DataFrame(columns=['Cleaned_Review', 'Sentiment', 'Category', 'Review_Length'])

    if request.method == 'POST':
        review = request.form.get('review')
        if review:
            cleaned_review = clean_text(review)
            X = vectorizer.transform([cleaned_review])
            prediction = model.predict(X)[0]
            category = classify_category(cleaned_review)
            review_length = len(cleaned_review.split())
            
            new_review = pd.DataFrame({
                'Cleaned_Review': [cleaned_review],
                'Sentiment': [prediction],
                'Category': [category],
                'Review_Length': [review_length]
            })
            user_reviews_df = pd.concat([user_reviews_df, new_review], ignore_index=True)
            try:
                user_reviews_df.to_csv(user_reviews_file, index=False)
                logger.info("Saved new review to user_reviews.csv")
            except Exception as e:
                logger.error(f"Error saving to {user_reviews_file}: {e}")
            
            for _, row in anomalies_df.iterrows():
                send_alert(row['Z_Score'], row['Batch'], 'batch')

    # Compute data for charts
    sentiment_counts = user_reviews_df['Sentiment'].value_counts().to_dict()
    categories = user_reviews_df['Category'].value_counts().to_dict()

    # Sentiment Trends
    sentiment_trends = {'labels': [], 'Positive': [], 'Neutral': [], 'Negative': []}
    if not user_reviews_df.empty:
        user_reviews_df['Submission'] = user_reviews_df.index
        submissions = user_reviews_df['Submission'].unique()
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            sentiment_df = user_reviews_df[user_reviews_df['Sentiment'] == sentiment]
            cumsum = sentiment_df.groupby('Submission').size().cumsum().reindex(submissions, fill_value=0)
            sentiment_trends[sentiment] = cumsum.tolist()
        sentiment_trends['labels'] = submissions.tolist()
    logger.info("Prepared sentiment_trends data")

    # Sentiment by Category
    sentiment_by_category = {'labels': [], 'Positive': [], 'Neutral': [], 'Negative': []}
    if not user_reviews_df.empty and 'Category' in user_reviews_df.columns:
        pivot = user_reviews_df.pivot_table(index='Category', columns='Sentiment', aggfunc='size', fill_value=0)
        sentiment_by_category['labels'] = pivot.index.tolist()
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            sentiment_by_category[sentiment] = pivot.get(sentiment, pd.Series(0, index=pivot.index)).tolist()
    logger.info("Prepared sentiment_by_category data")

    # Keyword Frequency
    keyword_frequency = {'labels': [], 'counts': []}
    if not user_reviews_df.empty:
        text = ' '.join(user_reviews_df['Cleaned_Review'].astype(str).str.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        keyword_counts = Counter(tokens)
        top_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        keyword_frequency['labels'] = list(top_keywords.keys())
        keyword_frequency['counts'] = list(top_keywords.values())
    logger.info("Prepared keyword_frequency data")

    # Review Length Distribution
    review_length_data = {'bins': [], 'counts': []}
    if not user_reviews_df.empty:
        bins = np.histogram(user_reviews_df['Review_Length'], bins=20)[1].astype(int)
        counts = np.histogram(user_reviews_df['Review_Length'], bins=20)[0].tolist()
        review_length_data['bins'] = bins[:-1].tolist()
        review_length_data['counts'] = counts
    logger.info("Prepared review_length_data")

    return render_template('index.html',
                          prediction=prediction,
                          category=category,
                          topics=topics_df.to_dict('records'),
                          anomalies=anomalies_df.to_dict('records'),
                          keywords=keywords_df.to_dict('records'),
                          sentiment_counts=sentiment_counts,
                          categories=categories,
                          sentiment_trends=sentiment_trends,
                          sentiment_by_category=sentiment_by_category,
                          keyword_frequency=keyword_frequency,
                          review_length_data=review_length_data)

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword').lower()
    if keyword:
        user_reviews_file = 'user_reviews.csv'
        if os.path.exists(user_reviews_file):
            try:
                user_reviews_df = pd.read_csv(user_reviews_file)
                filtered_reviews = user_reviews_df[user_reviews_df['Cleaned_Review'].str.contains(keyword, case=False, na=False)]
                logger.info(f"Found {len(filtered_reviews)} reviews for keyword '{keyword}'")
                return jsonify({'reviews': filtered_reviews[['Cleaned_Review', 'Sentiment', 'Category']].to_dict('records')})
            except Exception as e:
                logger.error(f"Error searching {user_reviews_file}: {e}")
                return jsonify({'reviews': []})
        logger.warning("No user_reviews.csv for search")
        return jsonify({'reviews': []})
    logger.warning("Empty keyword in search")
    return jsonify({'reviews': []})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)