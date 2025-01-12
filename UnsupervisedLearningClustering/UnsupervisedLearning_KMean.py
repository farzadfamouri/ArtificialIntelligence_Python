
# Unsupervised
# --------Clustering
# -------------------- K-Mean

from sklearn.cluster import KMeans

import pandas as pd

data = pd.read_csv("seeds-width-vs-length.csv")

print(data)

model = KMeans(n_clusters=3)
model.fit(data.iloc[:-5])

labels = model.predict(data.iloc[:-5])
print(labels)

print(model.predict(data[-4:]))

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=data.iloc[:-5, 0],  # First column
    y=data.iloc[:-5, 1],  # Second column
    hue=labels,      # Cluster labels for first part of data
    palette="viridis"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clustering")
plt.show()

# ----------------------Finding model centroids:

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y, marker="D", s=50)
plt.show()

# Evaluation

# --------------Cross Tabulation (Mostly for the categorical dataset)

# Add the cluster labels as a new column in the DataFrame
df2 = data.iloc[:-5].copy()  # Exclude the last 5 rows
df2['labels'] = labels  # Add labels as a new column

# Perform cross-tabulation on the labels and one of the features
# (For example, feature at column 0)
ct = pd.crosstab(df2['labels'], df2.iloc[:, 0])  # Replace column index as needed
print(ct)

df3 = df2.sort_values(by='labels')

print('For comparison purpose, df2: \n', df2, 'df3: \n', df3)

# --------------------------------Inertia

print("How good is the model by inertia criteria: \n", model.inertia_)


#---------------------------------Plot Inertia
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples

    model.fit(data.iloc[:-5])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


