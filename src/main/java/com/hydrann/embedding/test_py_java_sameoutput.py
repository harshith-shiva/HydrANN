from embedder import MiniLMEmbedder
import numpy as np 

embedder = MiniLMEmbedder()


text = "The quick brown fox jumps over the lazy dog."

# encoded = embedder.tokenizer(
#     text,
#     padding=False,
#     truncation=True,
#     return_tensors="np"
# )

# print("TOKENS:")
# print(embedder.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

# print("\nINPUT IDS:")
# print(encoded["input_ids"][0].tolist())

# print("\nATTENTION MASK:")
# print(encoded["attention_mask"][0].tolist())

# print("\nTYPE IDS:")
# print(encoded["token_type_ids"][0].tolist())

embedding = embedder.embed(text)

print("Python embedding shape:", embedding.shape)

print("\nFirst 10 Python values:")
print(embedding[0][:10])

np.savetxt(
    "python_embedding.txt",
    embedding[0]
)