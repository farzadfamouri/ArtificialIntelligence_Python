#Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from Unsupervised_KMean_Standardization import labels

data = pd.read_csv("seeds-width-vs-length.csv")

cleaned_data = data.iloc[ :-5].select_dtypes(include=[float, int]).dropna()

# Create a TSNE instance: model
model = TSNE(learning_rate = 200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(cleaned_data.values)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1st feature: ys
ys = tsne_features[:, 1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys, c =labels)
plt.show()