import pandas as pd

try:
    df = pd.read_csv('tripadvisor_hotel_reviews.csv', encoding='latin1')
    print("Dataset loaded successfully.")
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    df.to_pickle('raw_dataset.pkl')
except FileNotFoundError:
    print("Error: 'tripadvisor_hotel_reviews.csv' not found.")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)