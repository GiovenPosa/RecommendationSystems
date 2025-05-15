# Hybrid Recommendation System: Collaborative Filtering + Content-Based Filtering
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler

# Load datasets
column_names = ['User_ID', 'User_Names', 'Movie_ID', 'Rating', 'Timestamp']
movies_df = pd.read_csv('Extended_Movie_data.csv', sep=',', names=column_names)

movies_title_df = pd.read_csv("Extended_Movie_Id_Titles.csv")
movies_title_df.rename(columns={'item_id': 'Movie_ID', 'title': 'Movie_Title', 'genre': 'Genre'}, inplace=True)

movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")

# ================= Collaborative Filtering Setup =================
unique_users = sorted(movies_df.User_ID.unique())
unique_movies = sorted(movies_df.Movie_ID.unique())
user_id_map = {uid: i for i, uid in enumerate(unique_users)}
movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}
reverse_movie_id_map = {i: mid for mid, i in movie_id_map.items()}

n_users = len(unique_users)
n_movies = len(unique_movies)
ratings = np.zeros((n_users, n_movies))

for row in movies_df.itertuples():
    uid = user_id_map[row.User_ID]
    mid = movie_id_map[row.Movie_ID]
    ratings[uid, mid] = row.Rating

user_similarity = cosine_similarity(ratings)

# ================= Content-Based Filtering Setup =================
genre_global_frequency = Counter()
for genres in movies_df['Genre']:
    for genre in genres.split(','):
        genre_global_frequency[genre.strip()] += 1

def content_based_recommendation(user_id, top_n=10):
    user_ratings = movies_df[(movies_df['User_ID'] == user_id) & (movies_df['Rating'] >= 4)]
    if user_ratings.empty:
        return []

    user_genre_weight = Counter()
    for _, row in user_ratings.iterrows():
        weight = row['Rating']
        for genre in row['Genre'].split(','):
            genre = genre.strip()
            user_genre_weight[genre] += weight / genre_global_frequency[genre]

    rated_movie_ids = set(user_ratings['Movie_ID'])
    unique_movies_df = movies_df.drop_duplicates(subset=['Movie_ID'])
    scores = []

    for _, row in unique_movies_df.iterrows():
        if row['Movie_ID'] in rated_movie_ids:
            continue
        score = sum(user_genre_weight[g.strip()] for g in row['Genre'].split(',') if g.strip() in user_genre_weight)
        scores.append((row['Movie_ID'], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ================= Hybrid Recommendation =================
def hybrid_recommendation(user_id, top_n=10, cf_weight=0.4, cbf_weight=0.6):
    user_index = user_id_map.get(user_id)
    if user_index is None:
        print("User not found.")
        return

    user_name = movies_df[movies_df['User_ID'] == user_id]['User_Names'].iloc[0]

    # Collaborative Filtering Scores
    similarity_scores = user_similarity[user_index]
    weighted_scores = similarity_scores @ ratings
    similarity_sum = similarity_scores.sum()
    predicted_ratings = weighted_scores / (similarity_sum + 1e-8)
    already_rated = ratings[user_index] > 0
    predicted_ratings[already_rated] = 0

    cf_scores = [(reverse_movie_id_map[i], predicted_ratings[i]) for i in np.argsort(predicted_ratings)[::-1][:top_n]]
    cf_mids, cf_vals = zip(*cf_scores)
    cf_vals_scaled = MinMaxScaler().fit_transform(np.array(cf_vals).reshape(-1, 1)).flatten()
    cf_scores_scaled = list(zip(cf_mids, cf_vals_scaled))

    # Content-Based Filtering Scores
    cbf_scores = content_based_recommendation(user_id, top_n=top_n * 2)
    if cbf_scores:
        cbf_mids, cbf_vals = zip(*cbf_scores)
        cbf_vals_scaled = MinMaxScaler().fit_transform(np.array(cbf_vals).reshape(-1, 1)).flatten()
        cbf_dict_scaled = dict(zip(cbf_mids, cbf_vals_scaled))
    else:
        cbf_dict_scaled = {}

    # Combine scaled CF and CBF scores
    combined_scores = {}
    for mid, score in cf_scores_scaled:
        combined_scores[mid] = combined_scores.get(mid, 0) + cf_weight * score
    for mid, score in cbf_dict_scaled.items():
        combined_scores[mid] = combined_scores.get(mid, 0) + cbf_weight * score

    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    final_df = pd.DataFrame(sorted_combined, columns=['Movie_ID', 'Hybrid_Score'])

    # Normalize scores to percentages
    max_score = final_df['Hybrid_Score'].max()
    final_df['Score (%)'] = (final_df['Hybrid_Score'] / max_score) * 100
    final_df['Score (%)'] = final_df['Score (%)'].round(2)

    final_df = pd.merge(final_df, movies_df[['Movie_ID', 'Movie_Title', 'Genre']].drop_duplicates(), on='Movie_ID')
    print("Running Hybrid RECOMMENDATION SYSTEM")
    print(f"\nTop Recommendations for {user_name} (User ID: {user_id}):")
    print("------------------------------------------------------------")
    print(final_df[['Movie_Title', 'Genre', 'Score (%)']])

#RUN ENGINE
hybrid_recommendation(user_id=501, top_n=10, cf_weight=0.4, cbf_weight=0.6)
