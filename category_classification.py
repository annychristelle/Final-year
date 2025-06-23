import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import os

  # Ensure 'static' directory exists
if not os.path.exists('static'):
      os.makedirs('static')

try:
      df = pd.read_pickle('cleaned_dataset.pkl')
except FileNotFoundError:
      print("Error: 'cleaned_dataset.pkl' not found. Run clean_dataset.py first.")
      exit(1)

nlp = spacy.load('en_core_web_sm')
categories = {
      'Cleanliness': ['clean', 'dirty', 'hygiene', 'tidy'],
      'Service': ['staff', 'service', 'friendly', 'rude'],
      'Check-in': ['check-in', 'check in', 'arrival'],
      'Facilities': ['wifi', 'pool', 'gym', 'amenities']
  }

def classify_review(review):
      review_lower = str(review).lower()
      for category, keywords in categories.items():
          if any(keyword in review_lower for keyword in keywords):
              return category
      return 'Other'

df['Category'] = df['Cleaned_Review'].apply(classify_review)
df[['Cleaned_Review', 'Sentiment', 'Category']].to_csv('categorized_reviews.csv', index=False)

plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=df)
plt.title('Review Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Number of Reviews')
plt.savefig('static/category_distribution.png')
plt.close()
print("Category classification completed. 'categorized_reviews.csv' and 'static/category_distribution.png' created.")