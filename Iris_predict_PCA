import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Transform the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce dimensions to 3D for visualization
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Create a LogisticRegression instance
model = LogisticRegression()

# Fit the model to the PCA-reduced training data
model.fit(X_train_pca, y_train)

# Make predictions on the PCA-reduced test data
y_pred = model.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot decision boundaries in 3D using PCA reduced dimensions
def plot_decision_boundary_3d(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a dense mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30),
                             np.linspace(y_min, y_max, 30),
                             np.linspace(z_min, z_max, 30))
    
    # Predict on the mesh grid
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    ax.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, int(zz.shape[2]/2)], alpha=0.3, cmap=plt.cm.Paired)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolor='k', marker='o', cmap=plt.cm.Paired)
    
    # Labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Decision Boundary with PCA-reduced Dimensions')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Plot decision boundary in 3D with PCA-reduced dimensions
plot_decision_boundary_3d(model, X_test_pca, y_test)

# Show the relationship between PCA components and original features
pca_components = pd.DataFrame(pca.components_, columns=feature_names, index=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
print(pca_components)

# Visualize PCA components and original features relationship
plt.figure(figsize=(10, 7))
plt.imshow(pca_components.T, cmap='viridis', aspect='auto')
plt.colorbar()
plt.yticks(range(len(feature_names)), feature_names)
plt.xticks(range(len(pca_components.index)), pca_components.index, rotation=45)
plt.title('PCA Components and Original Features')
plt.xlabel('PCA Components')
plt.ylabel('Original Features')
plt.show()
