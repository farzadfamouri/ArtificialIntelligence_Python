from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
# Load the dataset
data = pd.read_csv('seeds-width-vs-length.csv')

#grains = pd.read_csv('seeds.csv')

width = data.iloc[:, 0]
length = data.iloc[:, 1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print("Correlation before Decorrelating: \n", correlation)

print("Mean before decorrelating: \n", data.mean())
model = PCA()
pca_features = model.fit_transform(data)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print("Correlation after Decorrelating: \n",correlation)

print("Components of the PCA model: \n", model.components_)
print("NUmber of Components of the PCA model: \n", model.n_components_)

# Finding intrinsic features:
features = range(model.n_components_)
plt.bar(features, model.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel("PCA feature")
plt.show()

print("Mean after decorrelating: \n", model.mean_)

# Display the first principal component( The arrow shows the direction of correlation:)

mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

plt.scatter(width, length)
plt.arrow(mean[0], mean[1],first_pc[0], first_pc[1], color='red', width=0.01)
plt.title("Correlation arrow on the original) dataset")
# Keep axes on same scale
plt.axis('equal')
plt.show()