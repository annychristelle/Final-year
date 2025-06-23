import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv('cleaned_tripadvisor_reviews.csv')

# Text length analysis
df['Review_Length'] = df['Cleaned_Review'].apply(lambda x: len(x.split()))
print("Review Length Statistics:")
print(df['Review_Length'].describe())

# Word frequency analysis
all_words = ' '.join(df['Cleaned_Review']).split()
word_freq = Counter(all_words)
print("\nTop 5 Common Words:")
print(word_freq.most_common(5))

# Sentiment distribution plot
plt.figure(figsize=(8, 6))
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.close()

# Word cloud for positive reviews
positive_reviews = ' '.join(df[df['Sentiment'] == 'Positive']['Cleaned_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.savefig('positive_wordcloud.png')
plt.close()