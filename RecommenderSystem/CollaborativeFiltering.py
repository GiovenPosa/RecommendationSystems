import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# Load the ratong data into Dataframe (DF)
column_names = ['User_ID', 'User_Names', 'Movie_ID', 'Rating', 'Timestamp']
movies_df = pd.read_csv('Extended_Movie_data.csv', sep =',', names = column_names)

# Load the movie information in the Dataframe
movies_title_df = pd.read_csv("Extended_Movie_Id_Titles.csv")
movies_title_df.rename(columns = {'item_id':'Movie_ID', 'title': 'Movie_Title'}, inplace = True)

# Merge the Dataframes
movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")

unique_users = sorted(movies_df.User_ID.unique())
unique_movies = sorted(movies_df.Movie_ID.unique())
n_users = len(unique_users)
n_movies = len(unique_movies)

user_id_map = {uid: i for i, uid in enumerate(unique_users)}
movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}
reverse_movie_id_map = {i: mid for mid, i in movie_id_map.items()}

ratings = np.zeros((n_users, n_movies))

for row in movies_df.itertuples():
  uid = user_id_map[row.User_ID]
  mid = movie_id_map[row.Movie_ID]
  ratings[uid, mid] = row.Rating

user_similarity = cosine_similarity(ratings)

def recommend_movies(user_index, ratings_matrix, similarity_matrix, top_n=10):
  similarity_scores = similarity_matrix[user_index]
  weighted_scores = similarity_scores @ ratings_matrix
  similarity_sums = similarity_scores.sum()
  
  # Avoid division by zero
  predicted_ratings = weighted_scores /(similarity_sums + 1e-8)
  
  # Dont recommend movies the user has already rated
  already_rated = ratings_matrix[user_index] > 0
  predicted_ratings[already_rated] = 0
  
  # Get the top N movie indices
  recommended_indices = np.argsort(predicted_ratings)[::-1][:top_n]
  
  return recommended_indices, predicted_ratings[recommended_indices]

def view_recommendations(user_index, movies_df, ratings_matrix, similarity_matrix, top_n=10):
  indices, scores = recommend_movies(user_index, ratings_matrix, similarity_matrix, top_n)
  recommend_movie_ids = [reverse_movie_id_map[i] for i in indices]
  
  original_user_id = unique_users[user_index]
  user_name = movies_df[movies_df['User_ID'] == original_user_id]['User_Names'].iloc[0]
  percent_scores = scores * 200
  # Create a Dataframe for matched Movie_ID and their predicted scores
  recommended_df = pd.DataFrame({ 'Movie_ID' : recommend_movie_ids, 'Score' : percent_scores })
  
  # Join with movie titles
  recommended_titles = pd.merge(recommended_df,
                                movies_df[['Movie_ID', 'Movie_Title']].drop_duplicates(),
                                on='Movie_ID',
                                how='left')
  
  
  print(f"\nTop {top_n} recommendations for User: {user_name} (User ID: {user_index + 1}):")
  print("---------------------------------------------------")
  for i, row in recommended_titles.iterrows():
    print(f"{row['Movie_Title']} -- {row['Score']:.2f}% Picked For You")
    
def run_recommender(user_index, top_n=5):
  print("Running COLLABORATIVE FILTERING RECOMMENDATION ENGINE")
  view_recommendations(user_index, movies_df, ratings, user_similarity, top_n)
  
run_recommender(user_index=500, top_n=10)
    
  