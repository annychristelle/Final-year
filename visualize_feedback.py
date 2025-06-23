import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

if not os.path.exists('static'):
    os.makedirs('static')
    logger.info("Created static directory")

# Load user reviews
user_reviews_file = 'user_reviews.csv'
if os.path.exists(user_reviews_file) and os.path.getsize(user_reviews_file) > 0:
    try:
        df = pd.read_csv(user_reviews_file)
        logger.info(f"Loaded {len(df)} reviews from {user_reviews_file}")
    except Exception as e:
        logger.error(f"Error loading {user_reviews_file}: {e}")
        df = pd.DataFrame(columns=['Cleaned_Review', 'Sentiment', 'Category', 'Review_Length'])
else:
    logger.warning("No user reviews found. Creating empty visualizations.")
    df = pd.DataFrame(columns=['Cleaned_Review', 'Sentiment', 'Category', 'Review_Length'])

# Sentiment Trends Over Time
plt.figure(figsize=(10, 6))
if not df.empty:
    df['Submission'] = df.index
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        sentiment_df = df[df['Sentiment'] == sentiment].groupby('Submission').size().cumsum()
        plt.plot(sentiment_df.index, sentiment_df, label=sentiment)
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Submission Number')
    plt.ylabel('Cumulative Count')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Sentiment Trends Over Time')
plt.savefig('static/sentiment_trends.png')
plt.close()
logger.info("Created sentiment_trends.png")

# Sentiment by Category
plt.figure(figsize=(10, 6))
if not df.empty and 'Category' in df.columns:
    sentiment_category = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_category.plot(kind='bar', stacked=True, color=['#f44336', '#ffeb3b', '#4caf50'])
    plt.title('Sentiment by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Sentiment by Category')
plt.xticks(rotation=45)
plt.savefig('static/sentiment_by_category.png')
plt.close()
logger.info("Created sentiment_by_category.png")

# Keyword Frequency Bar Chart
plt.figure(figsize=(8, 6))
if not df.empty:
    text = ' '.join(df['Cleaned_Review'].astype(str).str.lower())
    tokens = word_tokenize(text)
    keyword_counts = Counter(tokens)
    top_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    sns.barplot(x=list(top_keywords.keys()), y=list(top_keywords.values()))
    plt.title('Top 5 Keywords in User Reviews')
    plt.xlabel('Keyword')
    plt.ylabel('Frequency')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Top 5 Keywords in User Reviews')
plt.xticks(rotation=45)
plt.savefig('static/keyword_frequency.png')
plt.close()
logger.info("Created keyword_frequency.png")

# Review Length Distribution
plt.figure(figsize=(8, 6))
if not df.empty:
    sns.histplot(df['Review_Length'], bins=20, kde=True)
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length (Words)')
    plt.ylabel('Count')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Review Length Distribution')
plt.savefig('static/review_length_distribution.png')
plt.close()
logger.info("Created review_length_distribution.png")

# Review Length by Sentiment
plt.figure(figsize=(8, 6))
if not df.empty:
    sns.boxplot(x='Sentiment', y='Review_Length', data=df)
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
plt.title('Review Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Review Length (Words)')
plt.savefig('static/review_length_by_sentiment.png')
plt.close()
logger.info("Created review_length_by_sentiment.png")

# Sentiment Distribution
plt.figure(figsize=(8, 6))
if not df.empty:
    sns.countplot(x='Sentiment', data=df)
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('static/sentiment_distribution.png')
plt.close()
logger.info("Created sentiment_distribution.png")

# Category Distribution
if 'Category' in df.columns and not df.empty:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Category', data=df)
    plt.title('Review Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/category_distribution.png')
    plt.close()
    logger.info("Created category_distribution.png")
else:
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'No category data available', ha='center', va='center')
    plt.title('Review Distribution by Category')
    plt.savefig('static/category_distribution.png')
    plt.close()
    logger.info("Created category_distribution.png")

# Keyword Cloud
if not df.empty:
    text = ' '.join(df['Cleaned_Review'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
else:
    text = 'No data available'
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('static/keyword_cloud.png')
plt.close()
logger.info("Created keyword_cloud.png")

# Topic Distribution
if not df.empty:
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    try:
        X = vectorizer.fit_transform(df['Cleaned_Review'])
        nmf = NMF(n_components=min(5, X.shape[0]), random_state=42)
        nmf.fit(X)
        topic_weights = nmf.transform(X).sum(axis=0)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=np.arange(1, len(topic_weights) + 1), y=topic_weights)
        plt.title('Topic Distribution in User Reviews')
        plt.xlabel('Topic')
        plt.ylabel('Total Weight')
        plt.savefig('static/topics_distribution.png')
        plt.close()
        logger.info("Created topics_distribution.png")
    except ValueError as e:
        logger.warning(f"Topic modeling failed: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Insufficient data for topic modeling', ha='center', va='center')
        plt.title('Topic Distribution in User Reviews')
        plt.savefig('static/topics_distribution.png')
        plt.close()
        logger.info("Created topics_distribution.png (placeholder)")
else:
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Topic Distribution in User Reviews')
    plt.savefig('static/topics_distribution.png')
    plt.close()
    logger.info("Created topics_distribution.png (placeholder)")

# Negative Anomalies
if not df.empty:
    df['Batch'] = df.index // 10
    negative_counts = df[df['Sentiment'] == 'Negative'].groupby('Batch').size().reset_index(name='Negative_Count')
    if not negative_counts.empty:
        negative_counts['Z_Score'] = (negative_counts['Negative_Count'] - negative_counts['Negative_Count'].mean()) / negative_counts['Negative_Count'].std()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Batch', y='Negative_Count', size='Z_Score', data=negative_counts)
        plt.title('Negative Sentiment Anomalies')
        plt.xlabel('Batch')
        plt.ylabel('Negative Count')
        plt.savefig('static/negative_anomalies.png')
        plt.close()
        logger.info("Created negative_anomalies.png")
    else:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No negative reviews available', ha='center', va='center')
        plt.title('Negative Sentiment Anomalies')
        plt.savefig('static/negative_anomalies.png')
        plt.close()
        logger.info("Created negative_anomalies.png (placeholder)")
else:
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
    plt.title('Negative Sentiment Anomalies')
    plt.savefig('static/negative_anomalies.png')
    plt.close()
    logger.info("Created negative_anomalies.png (placeholder)")

logger.info("Visualizations created: 'static/sentiment_trends.png', 'static/sentiment_by_category.png', 'static/keyword_frequency.png', 'static/review_length_distribution.png', 'static/review_length_by_sentiment.png', 'static/sentiment_distribution.png', 'static/category_distribution.png', 'static/keyword_cloud.png', 'static/topics_distribution.png', 'static/negative_anomalies.png'")