import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
data = pd.read_csv('wikipedia-vectors.csv', header=None)

titles = data.iloc[0, 1:].to_list()
numeric_data = data.iloc[1:, 1:].astype(float).values.T

# Convert the numeric data to a CSR matrix
articles = csr_matrix(numeric_data)
#print(titles)
#print(articles)
# Import NMF


# Create an NMF instance: model
model = NMF(n_components =6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features =  model.transform(articles)

# Print the NMF features
print("model features: \n",nmf_features.round(2))
print("components of the first row: \n", model.components_[0])

df = pd.DataFrame(nmf_features, index=titles)

print(df.index)

print("Article Cristiano Ronaldo features: ", df.loc["Cristiano Ronaldo"])

print("Article shape: \n", articles.shape)
print("Components shape: \n", model.components_.shape)
print("Features shape: \n", nmf_features.shape)