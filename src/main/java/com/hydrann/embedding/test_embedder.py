from embedder import MiniLMEmbedder


embedder = MiniLMEmbedder()

paragraph_a = """
The performance of a machine learning system depends heavily on the data used to train it. By examining many examples during training, the model learns patterns and modifies its parameters to produce more accurate predictions. A sufficiently large, diverse, and reliable dataset helps the trained model generalize better and make accurate predictions on data it has never encountered before."""
paragraph_b = """
The performance of a machine learning system depends heavily on the data used to train it. By examining many examples during training, the model learns patterns and modifies its parameters to produce more accurate predictions. A sufficiently large, diverse, and reliable dataset helps the trained model generalize better and make accurate predictions on data it has never encountered before."""

embedding_a = embedder.embed(paragraph_a)
embedding_b = embedder.embed(paragraph_b)

similarity = embedder.cosine_similarity(
    embedding_a,
    embedding_b
)

print("Embedding A shape:", embedding_a.shape)
print("Embedding B shape:", embedding_b.shape)
print("Cosine similarity:", similarity)