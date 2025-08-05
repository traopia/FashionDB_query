import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests

# 1. Load data
df = pd.read_parquet("data/VogueRunway.parquet")
all_pics = pd.read_json("/Users/traopia/Documents/GitHub/Fashion-Patterns/Data/fashion_show_data_all_allpics.json", lines=True)


# Rename fashion_house to designer if it exists
if "fashion_house" in all_pics.columns:
    all_pics = all_pics.rename(columns={"fashion_house": "designer", "location":"city"})
    print("Renamed fashion_house to designer")
else:
    print("fashion_house column not found in all_pics")

# 2. Collection URL function
def create_collection_url(row):
    base_url = "https://www.vogue.com/fashion-shows/"
    season = str(row["season"]).lower()
    year = str(row["year"])
    category = str(row["category"]).lower() if pd.notna(row["category"]) and row["category"] and str(row["category"]).lower() != "nan" else None
    designer = str(row["designer"]).lower().replace(" ", "-")
    
    # Add city if available
    city = str(row["city"]).lower().replace(" ", "-") if pd.notna(row["city"]) and row["city"] and str(row["city"]).lower() != "nan" else None
    
    if pd.isna(category) or category is None or category == "nan":
        if city:
            return f"{base_url}{city}-{season}-{year}/{designer}"
        else:
            return f"{base_url}{season}-{year}/{designer}"
    else:
        if city:
            return f"{base_url}{city}-{season}-{year}-{category}/{designer}"
        else:
            return f"{base_url}{season}-{year}-{category}/{designer}"


import pandas as pd

# STEP 1: Explode image_urls to 1 row per URL in all_pics
all_pics_exploded = all_pics.explode('image_urls').rename(columns={'image_urls': 'url'})

# STEP 2: Create 'collection' column using your custom function
all_pics_exploded['collection'] = all_pics_exploded.apply(create_collection_url, axis=1)
df['collection'] = df.apply(create_collection_url, axis=1)

# STEP 3: Identify missing collections
df_collections = set(df['collection'])
all_pics_collections = set(all_pics_exploded['collection'])
missing_collections = all_pics_collections - df_collections

# STEP 4: Filter additional rows
additional_rows = all_pics_exploded[all_pics_exploded['collection'].isin(missing_collections)]

# STEP 5: Keep only the columns that exist in df
df_columns = df.columns.tolist()
additional_rows_filtered = additional_rows[list(set(df_columns).intersection(additional_rows.columns))]

# STEP 6: Generate new unique keys
if 'key' in df.columns:
    try:
        numeric_keys = pd.to_numeric(df['key'], errors='coerce')
        max_key = numeric_keys.max() if not numeric_keys.isna().all() else 0
    except Exception:
        max_key = len(df)
else:
    max_key = 0

new_keys = [str(i) for i in range(max_key + 1, max_key + 1 + len(additional_rows_filtered))]
additional_rows_filtered = additional_rows_filtered.copy()
additional_rows_filtered['key'] = new_keys

# STEP 7: Concatenate and ensure consistency
df_full = pd.concat([df, additional_rows_filtered], ignore_index=True)

# STEP 8: Add missing columns from df if any
for col in set(df.columns) - set(df_full.columns):
    df_full[col] = None

# STEP 9: Reorder columns to match df
df_full = df_full[df.columns.tolist()]
df_full.to_parquet("data/VogueRunway_full.parquet")





import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm

# Load existing image embeddings
embeddings = np.load("data/VogueRunway_image.npy")

# STEP 1: Filter rows to process
if len(additional_rows_filtered) > 0:
    print(f"Found {len(additional_rows_filtered)} additional rows to process")

    # Load the CLIP model that supports image inputs
    model = SentenceTransformer("clip-ViT-B-32")

    # Function to download and encode image
    def get_image_embedding_from_url(image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return model.encode(img, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding {image_url}: {e}")
            return None

    # STEP 2: Generate embeddings for each image
    new_embeddings = []
    for idx, row in tqdm(additional_rows_filtered.iterrows(), desc="Encoding new images", total=len(additional_rows_filtered)):
        image_url = row["url"] if "url" in row else None
        if image_url:
            emb = get_image_embedding_from_url(image_url)
            if emb is not None:
                new_embeddings.append(emb)
            else:
                new_embeddings.append(np.zeros(embeddings.shape[1]))
        else:
            print(f"No image URL found in row {idx}")
            new_embeddings.append(np.zeros(embeddings.shape[1]))

    # STEP 3: Stack with existing embeddings
    embeddings_full = np.vstack([embeddings, np.array(new_embeddings)])

    # Save embeddings
    np.save("data/VogueRunway_image_full.npy", embeddings_full)
    print("✅ Saved updated embeddings to data/VogueRunway_image_full.npy")

else:
    print("⚠️ No additional rows to process. Keeping original embeddings.")
    embeddings_full = embeddings





# # 4. Generate embeddings for new rows
# # Load existing embeddings
# embeddings = np.load("data/VogueRunway_image.npy")

# # Check if additional_rows has image URLs or paths
# if len(additional_rows) > 0:
#     print(f"Found {len(additional_rows)} additional rows to process")
    
#     # Load model
#     model = SentenceTransformer("clip-ViT-B-32")
    
#     # Function to get image embeddings from URL
#     def get_image_embedding_from_url(image_url):
#         import requests
#         from PIL import Image
#         from io import BytesIO
#         try:
#             response = requests.get(image_url, timeout=10)
#             response.raise_for_status()
#             img = Image.open(BytesIO(response.content))
#             return model.encode(img)
#         except Exception as e:
#             print(f"Error encoding {image_url}: {e}")
#             return None
    
#     # Generate embeddings for new images
#     new_embeddings = []
#     for idx, row in tqdm(additional_rows.iterrows(), desc="Encoding new images", total=len(additional_rows)):
#         try:
#             # Try to get image URL from the row
#             image_url = row.get('url', row.get('image_url', None))
#             if image_url:
#                 emb = get_image_embedding_from_url(image_url)
#                 if emb is not None:
#                     new_embeddings.append(emb)
#                 else:
#                     # Add zero vector if embedding failed
#                     new_embeddings.append(np.zeros(embeddings.shape[1]))
#             else:
#                 # Add zero vector if no image URL found
#                 new_embeddings.append(np.zeros(embeddings.shape[1]))
#         except Exception as e:
#             print(f"Error processing row {idx}: {e}")
#             new_embeddings.append(np.zeros(embeddings.shape[1]))
    
#     # Combine embeddings
#     if new_embeddings:
#         embeddings_full = np.vstack([embeddings, np.array(new_embeddings)])
#     else:
#         embeddings_full = embeddings
# else:
#     print("No additional rows found")
#     embeddings_full = embeddings

# # 5. Combine and save
# embeddings_full = np.vstack([embeddings, np.array(new_embeddings)])

# np.save("data/VogueRunway_image_full.npy", embeddings_full)
# print("Saved updated dataframe and embeddings.")