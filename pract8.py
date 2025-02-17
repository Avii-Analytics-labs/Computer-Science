import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Generate sample dataset with three natural clusters
np.random.seed(42)
n_samples = 300

# Create three clusters with different centers and spreads
cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, 2))
cluster2 = np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples//3, 2))
cluster3 = np.random.normal(loc=[2, -2], scale=0.5, size=(n_samples//3, 2))

# Combine clusters
X = np.vstack([cluster1, cluster2, cluster3])

def find_optimal_clusters(X, max_k=10):
    """
    Find optimal number of clusters using both elbow method and silhouette analysis
    """
    inertias = []
    silhouette_scores = []
    K = range(2, max_k+1)
    
    for k in K:
        # Create and fit model
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        
        # Get inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Elbow method plot
    plt.subplot(121)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Silhouette score plot
    plt.subplot(122)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    return K, inertias, silhouette_scores

def analyze_clusters(X, n_clusters):
    """
    Perform K-means clustering and analyze results
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Scatter plot of clusters
    plt.subplot(131)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.title('K-means Clustering Results')
    plt.colorbar(scatter)
    
    # Distribution of cluster assignments
    plt.subplot(132)
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    plt.bar(range(n_clusters), cluster_sizes)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title('Cluster Size Distribution')
    
    # Feature distributions by cluster
    plt.subplot(133)
    cluster_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    cluster_df['Cluster'] = labels
    
    # Calculate mean features by cluster
    cluster_means = cluster_df.groupby('Cluster').mean()
    
    # Create heatmap of cluster centers
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', center=0)
    plt.title('Cluster Centers Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    print("=" * 50)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        print(f"\nCluster {i}:")
        print(f"Number of points: {len(cluster_points)}")
        print(f"Cluster center: {centers[i]}")
        print(f"Standard deviation: {np.std(cluster_points, axis=0)}")
    
    return kmeans, labels, centers

# Find optimal number of clusters
print("Finding optimal number of clusters...")
K, inertias, silhouette_scores = find_optimal_clusters(X)

# Get optimal k from silhouette score
optimal_k = K[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Perform clustering with optimal k
print("\nPerforming K-means clustering...")
kmeans_model, cluster_labels, cluster_centers = analyze_clusters(X, optimal_k)

# Additional analysis: Cluster separation
def analyze_cluster_separation(X, labels, centers):
    """
    Analyze the separation between clusters
    """
    n_clusters = len(centers)
    distances = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distances[i,j] = np.linalg.norm(centers[i] - centers[j])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(distances, annot=True, cmap='YlOrRd')
    plt.title('Inter-cluster Distances')
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.show()
    
    return distances

print("\nAnalyzing cluster separation...")
cluster_distances = analyze_cluster_separation(X, cluster_labels, cluster_centers)
