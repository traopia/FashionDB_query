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


# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# import pandas as pd

# @st.cache_resource
# def load_text_model():
#     return SentenceTransformer("clip-ViT-B-32")

# def search_images_by_text(text, df_subset, all_embeddings, model, top_k=30):
#     """
#     Search the filtered df_subset using text prompt and slice relevant embeddings by index.
#     """
#     if df_subset.empty:
#         return df_subset

#     text_emb = model.encode([text], normalize_embeddings=True)
    
#     # Slice only the needed rows
#     indices = df_subset.index.values
#     filtered_embeddings = all_embeddings[indices]
    
#     # Use dot product instead of full cosine_similarity to save RAM
#     sims = filtered_embeddings @ text_emb.T
#     sims = sims.squeeze()  # shape: (n,)

#     top_indices = np.argsort(sims)[::-1][:top_k]
#     return df_subset.iloc[top_indices]

# def get_similar_images(df, image_index, embeddings, top_k=5):
#     """
#     Retrieve top_k most similar images to the one at image_index.
#     """
#     if image_index >= len(df):
#         return pd.DataFrame()  # fallback

#     query_emb = embeddings[image_index]
#     sims = embeddings @ query_emb.T
#     sims = sims.squeeze()

#     # Exclude the query image itself
#     top_indices = np.argsort(sims)[::-1]
#     top_indices = [i for i in top_indices if i != image_index][:top_k]
    
#     return df.iloc[top_indices]