import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Load the data
artists = pd.read_csv("scrobbler-small-sample.csv", header=0)
artist_names = pd.read_csv("artists.csv", header=None)

"""
# Inspect the structure of artist_names
print("Artist Names File:")
print(artist_names.head())
print("Shape of artist_names:", artist_names.shape)"""

# Use the index of artist_names as labels (assuming 1-based indexing for mapping)
artist_labels = artist_names.index + 1  # Map row numbers as labels
artist_names_list = artist_names[0].to_list()  # Artist names are in the only column

# Create a mapping of integers to artist names
artist_mapping = dict(zip(artist_labels, artist_names_list))

# Create a MaxAbsScaler
scaler = MaxAbsScaler()

# Create an NMF model
nmf = NMF(n_components=20)

# Create a Normalizer
normalizer = Normalizer()

# Create a pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists
norm_features = pipeline.fit_transform(artists)

# Map the row indices in the `artists` file to the corresponding artist names
artist_indices = artists.index + 1  # Assuming row numbers map to artist labels
artist_names_indexed = [artist_mapping.get(idx, f"Unknown Artist {idx}") for idx in artist_indices]

# Create a DataFrame with artist names as index
df = pd.DataFrame(norm_features, index=artist_names_indexed)

# Select the row of 'Bruce Springsteen'
try:
    artist = df.loc['Bruce Springsteen']
    # Compute cosine similarities
    similarities = df.dot(artist)
    # Display those with highest cosine similarity
    print(similarities.nlargest())
except KeyError:
    print("Artist 'Bruce Springsteen' not found in the data.")
