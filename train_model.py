import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import numpy as np
from keybert import KeyBERT

try:
    df = pd.read_pickle('cleaned_dataset.pkl')
except FileNotFoundError:
    print("Error: 'cleaned_dataset.pkl' not found. Run clean_dataset.py first.")
    exit(1)

# Sentiment Model
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Sentiment model accuracy: {accuracy:.2f}")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Topic Modeling
nmf = NMF(n_components=5, random_state=42)
nmf.fit(X)
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(nmf.components_):
    top_terms = [feature_names[i] for i in topic.argsort()[-10:]]
    topics.append({'Topic': f'Topic {topic_idx+1}', 'Terms': ', '.join(top_terms), 'Weight': nmf.transform(X)[:, topic_idx].sum()})
topics_df = pd.DataFrame(topics)
topics_df.to_csv('topics.csv', index=False)

# Topic-Specific Anomaly Detection
topic_assignments = nmf.transform(X).argmax(axis=1)
df['Topic'] = [f'Topic {i+1}' for i in topic_assignments]
negative_topic_counts = df[df['Sentiment'] == 'Negative'].groupby('Topic').size().reset_index(name='Negative_CountTopic')
negative_topic_counts['Z_Score'] = (negative_topic_counts['Negative_CountTopic'] - negative_topic_counts['Negative_CountTopic'].mean()) / negative_topic_counts['Negative_CountTopic'].std()
negative_topic_counts.to_csv('topic_anomalies.csv', index=False)

# Batch Anomaly Detection
df['Batch'] = df.index // 100
negative_counts = df[df['Sentiment'] == 'Negative'].groupby('Batch').size().reset_index(name='Negative_Count')
negative_counts['Z_Score'] = (negative_counts['Negative_Count'] - negative_counts['Negative_Count'].mean()) / negative_counts['Negative_Count'].std()
negative_counts.to_csv('anomalies.csv', index=False)

# Keyword Extraction with KeyBERT
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(' '.join(df['Cleaned_Review']), top_n=10)
keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
keywords_df.to_csv('keywords.csv', index=False)

print("Models trained and files saved: 'sentiment_model.pkl', 'tfidf_vectorizer.pkl', 'topics.csv', 'anomalies.csv', 'topic_anomalies.csv', 'keywords.csv'")