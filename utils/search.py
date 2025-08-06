from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Use a compatible CLIP model
model = SentenceTransformer("clip-ViT-B-32")

def search_images_by_text(text, df, embeddings, top_k=30):
    text_emb = model.encode([text])
    filtered_embeddings = embeddings[df.index]
    sims = cosine_similarity(text_emb, filtered_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return df.iloc[top_indices]

def get_similar_images(df, image_id, embeddings, top_k=5):
    index = int(image_id)  # adjust based on your ID setup
    query_emb = embeddings[index]
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_indices = np.argsort(sims)[::-1][1:top_k+1]
    return df.iloc[top_indices]


