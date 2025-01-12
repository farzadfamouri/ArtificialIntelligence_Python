import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
data = pd.read_csv('wikipedia-vectors.csv', header=None)

titles = data.iloc[0, 1:].to_list()
print(titles)
numeric_data = data.iloc[1:, 1:].astype(float).values.T

# Convert the numeric data to a CSR matrix
articles = csr_matrix(numeric_data)

nmf = NMF(n_components=7)

nmf_features = nmf.fit_transform(articles)
print(nmf_features.shape)
norm_features = normalize(nmf_features)

df = pd.DataFrame(norm_features, index=titles)
concurrent_article = df.loc["Cristiano Ronaldo"]
similarities = df.dot(concurrent_article)

print(similarities.nlargest())