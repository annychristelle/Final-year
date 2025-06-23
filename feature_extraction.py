from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.sparse

# Load cleaned dataset
df = pd.read_csv('cleaned_tripadvisor_reviews.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['Sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save feature names for inspection
feature_names = vectorizer.get_feature_names_out()
print("Sample TF-IDF Features:", feature_names[:10])

# Save train/test splits
scipy.sparse.save_npz('X_train.npz', X_train)
scipy.sparse.save_npz('X_test.npz', X_test)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)