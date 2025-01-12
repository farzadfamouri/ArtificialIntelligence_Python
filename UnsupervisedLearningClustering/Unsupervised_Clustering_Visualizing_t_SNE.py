# Import necessary libraries
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('company_stock_movements.csv')

# Separate company names and numeric data
companies = data['Company']  # Assuming the column with company names is called 'Company'
numeric_data = data.select_dtypes(include=['float', 'int'])  # Select only numeric columns

# Normalize the numeric data
normalizer = Normalizer()
normalized_movements = normalizer.fit_transform(numeric_data)

# Create a TSNE instance
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements
tsne_features = model.fit_transform(normalized_movements)

# Extract the 0th and 1st t-SNE features
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points with company names
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.show()