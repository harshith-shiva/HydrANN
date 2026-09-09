import numpy as np
from pathlib import Path

# Directory containing this Python script
BASE_DIR = Path(__file__).parent

python_embedding = np.loadtxt(
    BASE_DIR / "python_embedding.txt"
)

java_embedding = np.loadtxt(
    BASE_DIR / "java_embedding.txt"
)

cosine_similarity = np.dot(
    python_embedding,
    java_embedding
) / (
    np.linalg.norm(python_embedding)
    * np.linalg.norm(java_embedding)
)

max_difference = np.max(
    np.abs(python_embedding - java_embedding)
)

mean_difference = np.mean(
    np.abs(python_embedding - java_embedding)
)

print("Cosine similarity:", cosine_similarity)
print("Maximum absolute difference:", max_difference)
print("Mean absolute difference:", mean_difference)