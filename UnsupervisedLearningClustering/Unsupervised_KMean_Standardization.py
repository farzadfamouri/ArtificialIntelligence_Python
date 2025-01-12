from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv("seeds-width-vs-length.csv")
print(data)

scaler= StandardScaler()

# instead of StandardScaler, normalizer can be used:
"""
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()
"""
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(data.iloc[:-5])
labels = pipeline.predict(data.iloc[:-5])


print("How good is the model by inertia criteria: \n", kmeans.inertia_)