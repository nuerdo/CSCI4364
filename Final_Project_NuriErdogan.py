from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/nurie/Desktop/GWU/CSCI4364/Final_Project/HealthIndicators.csv",converters={"label":int})
X3 = df.to_numpy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X2 = scaler.fit_transform(X3)

#build the clustering model
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X2)

from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to health data
pca.fit(X2)

# transform data onto the first two principal components
X = pca.transform(X2)

# plot first vs. second principal component, colored by class
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(6), df.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
print("Cluster memberships:\n", y_kmeans)
