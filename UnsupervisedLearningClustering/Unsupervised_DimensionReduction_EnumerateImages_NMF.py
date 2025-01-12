import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
data = pd.read_csv('lcd_digits.csv')

# Select the 0th row: digit
digit = data.iloc[0]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.values.reshape((13,8))

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components = 7)

# Apply fit_transform to samples: features
features = model.fit_transform(data)


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0, :]

# Print digit_features
print(digit_features)


# !Warning! PCA is not cable reprsenting meaningful parts of images:
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model2 = PCA( n_components = 7)

# Apply fit_transform to samples: features
features = model2.fit_transform(data)

# Call show_as_image on each component
for component in model2.components_:
    show_as_image(component)