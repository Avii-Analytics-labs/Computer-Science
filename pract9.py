import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample high-dimensional dataset
np.random.seed(42)
n_samples = 1000
n_features = 10

# Create correlated features
X = np.random.randn(n_samples, n_features)
# Add some correlation between features
X[:, 1] = X[:, 0] * 0.7 + np.random.randn(n_samples) * 0.3
X[:, 2] = X[:, 0] * -0.5 + X[:, 1] * 0.8 + np.random.randn(n_samples) * 0.2

def perform_pca_analysis(X, n_components=None):
    """
    Perform PCA analysis with comprehensive evaluation
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Scree plot
    plt.subplot(131)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             explained_variance_ratio, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    
    # Cumulative variance plot
    plt.subplot(132)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio, 'ro-')
    plt.axhline(y=0.9, color='k', linestyle='--', label='90% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    
    # Heatmap of component loadings
    plt.subplot(133)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=[f'Feature {i+1}' for i in range(X.shape[1])]
    )
    sns.heatmap(loadings, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('PCA Component Loadings')
    
    plt.tight_layout()
    plt.show()
    
    # Print variance explained by each component
    print("\nExplained Variance by Component:")
    print("=" * 50)
    for i, var in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)")
    
    return pca, X_pca, loadings

def visualize_reduced_data(X_pca, n_components=2):
    """
    Visualize the data in reduced dimensional space
    """
    if n_components >= 2:
        plt.figure(figsize=(15, 5))
        
        # 2D scatter plot
        plt.subplot(131)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('First Two Principal Components')
        
        # Density plot
        plt.subplot(132)
        sns.kdeplot(data=pd.DataFrame(X_pca[:, :2], 
                                     columns=['PC1', 'PC2']))
        plt.title('Density Plot of First Two PCs')
        
        if n_components >= 3:
            # 3D scatter plot
            ax = plt.subplot(133, projection='3d')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                               c=X_pca[:, 2], cmap='viridis')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.colorbar(scatter)
            plt.title('First Three Principal Components')
        
        plt.tight_layout()
        plt.show()

def analyze_feature_contributions(loadings):
    """
    Analyze and visualize feature contributions to principal components
    """
    plt.figure(figsize=(12, 6))
    
    # Feature importance for first two PCs
    plt.subplot(121)
    feature_importance = loadings[['PC1', 'PC2']].abs()
    feature_importance.plot(kind='bar')
    plt.title('Feature Contributions to PC1 and PC2')
    plt.xticks(rotation=45)
    
    # Correlation between original features and PCs
    plt.subplot(122)
    sns.heatmap(loadings.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between PCs')
    
    plt.tight_layout()
    plt.show()

# Perform PCA analysis
print("Performing PCA analysis...")
pca_model, X_transformed, component_loadings = perform_pca_analysis(X)

# Visualize the reduced dimensional data
print("\nVisualizing reduced dimensional data...")
visualize_reduced_data(X_transformed, n_components=3)

# Analyze feature contributions
print("\nAnalyzing feature contributions...")
analyze_feature_contributions(component_loadings)

# Calculate optimal number of components for 90% variance
n_components_90 = np.argmax(np.cumsum(pca_model.explained_variance_ratio_) >= 0.9) + 1
print(f"\nNumber of components needed for 90% variance: {n_components_90}")

# Perform PCA with optimal components
print("\nPerforming PCA with optimal number of components...")
pca_optimal, X_optimal, loadings_optimal = perform_pca_analysis(X, n_components=n_components_90)
