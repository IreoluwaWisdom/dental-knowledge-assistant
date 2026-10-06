from rag import chunk_texts, get_embeddings
import json

embeddings = get_embeddings(chunk_texts)

with open("chunk_embeddings.json", "w") as file:
    json.dump(embeddings, file)

print(f"Saved {len(embeddings)} chunk embeddings.")