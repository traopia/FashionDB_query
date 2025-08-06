import streamlit as st
import pandas as pd
import numpy as np
from utils.search import search_images_by_text, get_similar_images

# Initialize session state
if 'show_similar_page' not in st.session_state:
    st.session_state.show_similar_page = False
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

# Load metadata and embeddings
# @st.cache_data
# def load_data():
#     # Load the enhanced dataset with additional rows from all_pics
#     df = pd.read_parquet("https://huggingface.co/datasets/traopia/vogue-runway/resolve/main/VogueRunway_full.parquet")
#     embeddings = np.load("https://huggingface.co/datasets/traopia/vogue-runway/blob/main/VogueRunway_image_full.npy")
#     return df, embeddings

import requests
from io import BytesIO
@st.cache_data
def load_data():
    # Load the Parquet file directly from Hugging Face
    df_url = "https://huggingface.co/datasets/traopia/vogue-runway/resolve/main/VogueRunway_full.parquet"
    df = pd.read_parquet(df_url)

    # Load the .npy file using requests
    npy_url = "https://huggingface.co/datasets/traopia/vogue-runway/resolve/main/VogueRunway_image_full.npy"
    response = requests.get(npy_url)
    response.raise_for_status()  # Raise error if download fails
    embeddings = np.load(BytesIO(response.content))

    return df, embeddings

# from huggingface_hub import hf_hub_download
# @st.cache_data(show_spinner="Loading FashionDB...")
# def load_data():
#     meta_path = hf_hub_download(repo_id="traopia/vogue-runway", filename="VogueRunway_full.parquet")
#     emb_path = hf_hub_download(repo_id="traopia/vogue-runway", filename="VogueRunway_image_full.npy")
#     df = pd.read_parquet(meta_path)
#     embeddings = np.load(emb_path, mmap_mode='r')
#     return df, embeddings

df, embeddings = load_data()

st.title("FashionDB Explorer")

# --- Filter Section ---
with st.sidebar:
    st.header("Filters")
    fashion_house = st.multiselect("Fashion House", options=df['designer'].unique())
    category = st.multiselect("Category", options=df['category'].unique())
    season = st.multiselect("Season", options=df['season'].unique())
    year_range = st.slider("Year", int(df['year'].min()), int(df['year'].max()), (2000, 2025))
    #designer = st.multiselect("Designer", options=df['fashion_designer'].unique())
    #birth_year = st.slider("Designer Year of Birth", 1900, 2020, (1950, 2000))

# Apply filters
filtered_df = df.copy()
if fashion_house:
    filtered_df = filtered_df[filtered_df['designer'].isin(fashion_house)]
if category:
    filtered_df = filtered_df[filtered_df['category'].isin(category)]
if season:
    filtered_df = filtered_df[filtered_df['season'].isin(season)]
filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
# filtered_df = filtered_df[(filtered_df['designer_birth_year'] >= birth_year[0]) & (filtered_df['designer_birth_year'] <= birth_year[1])]
# if designer:
#     filtered_df = filtered_df[filtered_df['fashion_designer'].isin(designer)]

# --- Text Query Section ---
query = st.text_input("Search (e.g., 'pink dress')")

if query:
    results = search_images_by_text(query, filtered_df, embeddings)
else:
    results = filtered_df.head(30)

def display_similar_images_page(selected_row, df, embeddings):
    """Display the similar images page"""
    st.title("🔍 Similar Images Explorer")
    
    # Back button
    if st.button("← Back to Search"):
        st.session_state.show_similar_page = False
        st.session_state.selected_image = None
        st.rerun()
    
    # Display the query image
    st.header("Query Image")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(selected_row['url'], caption=selected_row['designer'], use_container_width=True)
    
    with col2:
        # Show metadata for query image
        st.markdown("**📋 Query Image Metadata:**")
        metadata_cols = ['designer', 'season', 'year', 'category']
        available_cols = [col for col in metadata_cols if col in selected_row.index]
        
        if available_cols:
            for col in available_cols:
                if pd.notna(selected_row[col]):
                    st.markdown(f"**{col.title()}:** {selected_row[col]}")
            
            # Add collection URL if available
            if 'collection' in selected_row.index and pd.notna(selected_row['collection']):
                st.markdown("**Collection:**")
                st.markdown(f"[View on Vogue]({selected_row['collection']})")
        else:
            st.info("No metadata available for this item")
    
    # Get and display similar images
    st.header("Similar Images")
    similar_images = get_similar_images(df, selected_row['key'], embeddings, top_k=5)
    
    # Display similar images in a grid
    cols = st.columns(5)
    for i, (_, similar_row) in enumerate(similar_images.iterrows()):
        with cols[i % 5]:
            st.image(similar_row['url'], caption=similar_row['designer'], use_container_width=True)
            
            # Show metadata button for similar images
            if st.button(f"Show metadata {i}", key=f"similar_meta_{i}"):
                metadata_cols = ['designer', 'season', 'year', 'category']
                available_cols = [col for col in metadata_cols if col in similar_row.index]
                
                if available_cols:
                    st.markdown("**📋 Metadata:**")
                    for col in available_cols:
                        if pd.notna(similar_row[col]):
                            st.markdown(f"**{col.title()}:** {similar_row[col]}")
                    
                    # Add collection URL if available
                    if 'collection' in similar_row.index and pd.notna(similar_row['collection']):
                        st.markdown("**Collection:**")
                        st.markdown(f"[View on Vogue]({similar_row['collection']})")
                else:
                    st.info("No metadata available for this item")
            
            # Find similar button for similar images
            if st.button(f"Find similar {i}", key=f"similar_similar_{i}"):
                st.session_state.selected_image = similar_row
                st.rerun()

# --- Display Results ---
if st.session_state.show_similar_page and st.session_state.selected_image is not None:
    display_similar_images_page(st.session_state.selected_image, df, embeddings)
else:
    st.subheader("Results")
    cols = st.columns(5)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 5]:
            st.image(row['url'], caption=row['designer'], use_container_width=True)
            if st.button(f"metadata {i}"):
                # Create a nice metadata display with only specific columns
                metadata_cols = ['designer', 'season', 'year', 'category']
                available_cols = [col for col in metadata_cols if col in row.index]
                
                if available_cols:
                    for col in available_cols:
                        if pd.notna(row[col]):  # Only show non-null values
                            st.markdown(f"**{col.title()}:** {row[col]}")
                    
                    # Add collection URL if available
                    if 'collection' in row.index and pd.notna(row['collection']):
                        st.markdown("**Collection:**")
                        st.markdown(f"[View on Vogue]({row['collection']})")
                else:
                    st.info("No metadata available for this item")
            if st.button(f"find similar {i}"):
                st.session_state.show_similar_page = True
                st.session_state.selected_image = row
                st.rerun()