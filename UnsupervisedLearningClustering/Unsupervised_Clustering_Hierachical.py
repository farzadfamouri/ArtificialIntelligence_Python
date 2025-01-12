from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd

data = pd.read_csv("eurovision_2016.csv")


# Normalize the movements: normalized_movements
#normalized_data= normalize(data["Jury A"].iloc[1:27].values.reshape(-1, 1))

# Calculate the linkage: mergings
#mergings = linkage(normalized_data, method='complete')

mergings = linkage(data["Jury A"].iloc[1:27].values.reshape(-1, 1), method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=data["To country"].iloc[1:27].values, leaf_rotation=90, leaf_font_size=15)
plt.show()

from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergings, 15, criterion='distance')
print(labels)

import pandas as pd
eurovision = pd.DataFrame({'labels': labels, 'countries' : data["To country"].iloc[1:27]})

print(eurovision.sort_values(('labels')))