import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Example embeddings (replace these with your actual embeddings)
# Assuming each set of embeddings is a 2D NumPy array with shape (num_samples, embedding_dim)
# embeddings_set1 = np.random.rand(100, 50)  # 100 samples with 50-dimensional embeddings
# embeddings_set2 = np.random.rand(100, 50)  # Another 100 samples with 50-dimensional embeddings
with open("/work/yuxiang1234/Languagecodec/common_voice-ecapa-tcnn.pkl", "rb") as f: 
	embeddings = pickle.load(f)

	embeddings_set1 = embeddings.values()
	embeddings_set1 = [e.flatten() for e in embeddings_set1]

with open("/work/yuxiang1234/Languagecodec/hy-ecapa-tcnn.pkl", "rb") as f: 
	embeddings = pickle.load(f)

	embeddings_set2 = embeddings.values()
	embeddings_set2 = [e.flatten() for e in embeddings_set2][:1]
# Combine embeddings for t-SNE
embeddings_combined = np.vstack((embeddings_set1, embeddings_set2))

# Create labels to differentiate the sets
labels = np.array([0] * len(embeddings_set1) + [1] * len(embeddings_set2))

# Initialize t-SNE with 2 components for 2D visualization
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the combined embeddings
reduced_embeddings = tsne.fit_transform(embeddings_combined)

# Create a scatter plot
plt.figure(figsize=(10, 7))

# Plot embeddings set 1
plt.scatter(reduced_embeddings[labels == 0, 0], reduced_embeddings[labels == 0, 1], 
            c='blue', label='Set 1', marker='o', edgecolor='k')

# Plot embeddings set 2
plt.scatter(reduced_embeddings[labels == 1, 0], reduced_embeddings[labels == 1, 1], 
            c='red', label='Set 2', marker='^', edgecolor='k')

# Add title, labels, and legend
plt.title("t-SNE Visualization of Two Sets of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

# Show plot
plt.show()
plt.savefig("tsne.jpg")
