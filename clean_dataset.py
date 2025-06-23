import pandas as pd
from langdetect import detect
import re

try:
       df = pd.read_pickle('raw_dataset.pkl')
except FileNotFoundError:
       print("Error: 'raw_dataset.pkl' not found. Run load_dataset.py first.")
       exit(1)

def is_english(text):
       try:
           return detect(text) == 'en'
       except:
           return False

df['is_english'] = df['Review'].apply(is_english)
df = df[df['is_english']].copy()
df['Cleaned_Review'] = df['Review'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
df['Sentiment'] = df['Rating'].apply(lambda x: 'Positive' if x >= 4 else 'Neutral' if x == 3 else 'Negative')
df['Review_Length'] = df['Cleaned_Review'].apply(lambda x: len(x.split()))
df[['Cleaned_Review', 'Sentiment', 'Review_Length']].to_csv('cleaned_tripadvisor_reviews.csv', index=False)
df.to_pickle('cleaned_dataset.pkl')
print("Dataset cleaned successfully. 'cleaned_tripadvisor_reviews.csv' and 'cleaned_dataset.pkl' created.")