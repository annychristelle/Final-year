import pandas as pd
import spacy
from spacy.training import Example
import random
import joblib

try:
    df = pd.read_pickle('cleaned_dataset.pkl')
except FileNotFoundError:
    print("Error: 'cleaned_dataset.pkl' not found. Run clean_dataset.py first.")
    exit(1)

# Define categories
categories = ['Cleanliness', 'Service', 'Check-in', 'Facilities']

# Add Category column if missing
if 'Category' not in df.columns:
    def assign_category(review):
        review_lower = str(review).lower()
        keyword_map = {
            'Cleanliness': ['clean', 'dirty', 'hygiene', 'tidy'],
            'Service': ['staff', 'service', 'friendly', 'rude'],
            'Check-in': ['check-in', 'check in', 'arrival'],
            'Facilities': ['wifi', 'pool', 'gym', 'amenities']
        }
        for category, keywords in keyword_map.items():
            if any(keyword in review_lower for keyword in keywords):
                return category
        return 'Other'
    df['Category'] = df['Cleaned_Review'].apply(assign_category)

df = df[df['Category'].isin(categories)]  # Exclude 'Other'

# Prepare spaCy training data
nlp = spacy.blank('en')
textcat = nlp.add_pipe('textcat')
for category in categories:
    textcat.add_label(category)

train_data = []
for _, row in df.iterrows():
    cats = {cat: 1 if cat == row['Category'] else 0 for cat in categories}
    train_data.append((row['Cleaned_Review'], {'cats': cats}))

# Train model
optimizer = nlp.begin_training()
for _ in range(10):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
    print(f"Losses: {losses}")

# Save model
joblib.dump(nlp, 'category_classifier.pkl')
print("Category classifier saved: 'category_classifier.pkl'")

# Update cleaned_dataset.pkl and categorized_reviews.csv
df['Category'] = df['Cleaned_Review'].apply(lambda x: max(nlp(x).cats, key=lambda c: nlp(x).cats[c]))
df.to_pickle('cleaned_dataset.pkl')
df[['Cleaned_Review', 'Sentiment', 'Category']].to_csv('categorized_reviews.csv', index=False)
print("Updated 'cleaned_dataset.pkl' and 'categorized_reviews.csv' with spaCy predictions")