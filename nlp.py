import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Function to remove English stopwords (modify if using another language)
def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) # Change 'english' to 'german' if your text is in German
    words = text.split() 
    clean_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(clean_words)

# Load data
df = pd.read_csv("myntra.csv")
df.fillna("None", inplace=True)

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

# Encode the descriptions after removing stopwords
df['DescriptionVector'] = df['name'].apply(lambda x: model.encode(remove_stopwords(x)))

# Convert the list of vectors to a 2D array
vectors = np.array(df['DescriptionVector'].tolist())

# Initialize and fit NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
neigh.fit(vectors)

# Encode the input keyword after removing stopwords
input_keyword = "Blue Shoes"
vector_of_input_keyword = model.encode(remove_stopwords(input_keyword))

# Find the nearest neighbors
distances, indices = neigh.kneighbors([vector_of_input_keyword])

# Print out the nearest neighbors
for i in indices[0]:
    product_name = df.iloc[i]['ProductName']  # Assuming 'ProductName' is a column in your dataframe
    description = df.iloc[i]['name']

    print(f"Product Name: {product_name}")
    print(f"Description: {description}")
    print("-" * 40)
