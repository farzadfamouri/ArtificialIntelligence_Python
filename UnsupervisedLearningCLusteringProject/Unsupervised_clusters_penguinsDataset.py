# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca= PCA()

df = penguins_df.iloc[:, 0:3]
pca.fit(df)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.show()

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of PCA')
plt.show()

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples

    model.fit(reduced_data)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

from sklearn.pipeline import make_pipeline
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(reduced_data)
labels = pipeline.predict(reduced_data)
print(labels)

numeric_data = penguins_df.select_dtypes(include='number')

df2 = numeric_data.copy()  # Exclude the last 5 rows
df2['labels'] = labels  # Add labels as a new column

print(df2)

stat_penguins = pd.DataFrame()
stat_penguins = df2.groupby('labels').mean()
print(stat_penguins)