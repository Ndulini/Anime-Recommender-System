import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise import dump

# Load data
@st.cache_data
def load_anime():
    try:
        return pd.read_csv('anime.csv')
    except FileNotFoundError:
        st.error("anime.csv not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame() # Return empty dataframe on error

@st.cache_data
def load_train():
    try:
        # Use the cleaned train data if available, otherwise load the original
        # Assuming train_clean.csv might exist from previous steps, if not, use train.csv
        # If train_clean was only a dataframe in memory, stick to train.csv or adapt
        return pd.read_csv('train.csv') # Or 'train_clean.csv' if saved
    except FileNotFoundError:
        st.error("train.csv not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame() # Return empty dataframe on error


# Load the trained SVD model using st.cache_resource
@st.cache_resource
def load_svd_model():
    try:
        # Use the path where you saved your model
        model_path = 'svd_model'
        model, _ = dump.load(model_path)
        st.success(f"SVD model loaded successfully from {model_path}.")
        return model
    except FileNotFoundError:
        st.error(f"SVD model file '{model_path}' not found. Please train and save the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading SVD model: {e}")
        return None


anime = load_anime()
train = load_train()
model = load_svd_model() # Load the model

# Load cosine similarity matrix
# You need to re-calculate or load this here if you use content-based filtering
# For now, leaving the placeholder or implementing actual loading
# If content-based relies on external files (like TF-IDF matrix), load them here too
# from sklearn.metrics.pairwise import cosine_similarity # Import if needed
# from sklearn.feature_extraction.text import TfidfVectorizer # Import if needed
# Assuming you might need TF-IDF and cosine_sim for the content-based part
# @st.cache_resource # Cache the TF-IDF vectorizer and cosine similarity matrix
# def prepare_content_based_data(anime_df):
#     if anime_df.empty:
#          return None, None, None
#     anime_df = anime_df.copy() # Work on a copy
#     anime_df['genre'] = anime_df['genre'].fillna('Unknown').replace('Unknown', '')
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     title_to_index = pd.Series(anime_df.index, index=anime_df['name'])
#     return cosine_sim, title_to_index, anime_df

# cosine_sim, title_to_index, processed_anime = prepare_content_based_data(anime)


# Content-based recommender
# Update to use cached data if implemented above
def get_similar_anime(title, n=10):
     # This function needs the cosine_sim matrix and title_to_index mapping
     # Ensure these are loaded or computed within the Streamlit app context
     st.warning("Content-Based Filtering is not fully implemented in this app. Cosine similarity matrix is a placeholder.")
     # Placeholder implementation (replace with actual logic if needed)
     if anime.empty:
         return "Anime data not loaded."
     # Re-create title_to_index if not using cached version
     title_to_index = pd.Series(anime.index, index=anime['name'])
     idx = title_to_index.get(title)
     if idx is None:
         return f"'{title}' not found in the dataset."
     # Assuming cosine_sim is a global placeholder or loaded
     if cosine_sim is None: # Check if content-based data preparation failed
          return "Content-based data not available."
     try:
         sim_scores = list(enumerate(cosine_sim[idx]))
         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
         sim_scores = sim_scores[1:n+1]
         anime_indices = [i[0] for i in sim_scores]
         # Use the original anime df for results
         return anime[['name', 'genre']].iloc[anime_indices]
     except IndexError:
         return "Error processing similarity scores."


# Collaborative filtering recommender
def recommend_collaborative(user_id, n=10):
    if model is None or train.empty or anime.empty:
        return "Collaborative filtering is not available (model or data not loaded)."

    anime_ids = anime['anime_id'].unique()
    # Ensure user_id is in the training data before filtering
    if user_id not in train['user_id'].values:
         return f"User ID {user_id} not found in training data."

    rated = train[train['user_id'] == user_id]['anime_id'].tolist()
    preds = []
    for aid in anime_ids:
        if aid not in rated:
            try:
                # Ensure anime_id exists in the model's item mapping if necessary
                # Surprise handles unseen items by default, but predicting for an ID not in
                # the training set might give a default prediction.
                pred = model.predict(user_id, aid)
                preds.append((aid, pred.est))
            except Exception as e:
                st.warning(f"Could not predict for anime_id {aid}: {e}")
                continue # Skip this anime if prediction fails


    if not preds:
         return f"No unseen anime found for user {user_id}."

    preds.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i[0] for i in preds[:n]]
    # Use the original anime df for results
    return anime[anime['anime_id'].isin(top_ids)][['name', 'genre']]

# Streamlit UI
st.title("Anime Recommender System")

option = st.selectbox("Choose a recommender type:", ("Content-Based Filtering", "Collaborative Filtering"))

if option == "Content-Based Filtering":
    st.info("Content-Based Filtering is a placeholder implementation.")
    anime_title = st.text_input("Enter an anime title:")
    if anime_title:
        recommendations = get_similar_anime(anime_title)
        if isinstance(recommendations, str):
            st.warning(recommendations)
        elif not recommendations.empty:
            st.write("Top recommendations:")
            st.dataframe(recommendations)
        else:
             st.info("No similar anime found or data not available.")


elif option == "Collaborative Filtering":
    user_id_input = st.text_input("Enter your User ID:")
    if user_id_input:
        try:
            user_id = int(user_id_input)
            if model: # Check if model was loaded successfully
                recommendations = recommend_collaborative(user_id)
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                elif not recommendations.empty:
                    st.write("Top recommendations:")
                    st.dataframe(recommendations)
                else:
                    st.info("No recommendations found for this user.")
            else:
                 st.warning("SVD model not loaded. Cannot provide collaborative recommendations.")
        except ValueError:
            st.warning("Please enter a valid integer User ID.")
