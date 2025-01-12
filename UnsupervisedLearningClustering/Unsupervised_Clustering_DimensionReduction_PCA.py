from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd



data = pd.read_csv('fish.csv')

# Create scaler: scaler
scaler = StandardScaler()

scaled_samples = scaler.fit_transform(data.iloc[:, 1:])
# Create a PCA model with 2 components: pca
pca = PCA(n_components =2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)