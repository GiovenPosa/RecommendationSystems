from collections import Counter, defaultdict
import pandas as pd

column_names = ['User_ID', 'User_Names', 'Movie_ID', 'Rating']
movies_df = pd.read_csv('Extended_Movie_data.csv', sep=',', names=column_names)

movies_title_df = pd.read_csv("Extended_Movie_Id_Titles.csv")
movies_title_df.rename(columns={'item_id': 'Movie_ID', 'title': 'Movie_Title', 'genre' : 'Genre'}, inplace=True)

movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")

# calculate global genre frequency
genre_global_frequency = Counter()
for genres in movies_df['Genre']:
  for genre in genres.split(','):
    genre_global_frequency[genre.strip()] += 1

def content_based_recommendation(user_id, movies_df, top_n=10):
  # Filter high-rated movies by the user 
  user_ratings = movies_df[(movies_df['User_ID'] == user_id) & (movies_df['Rating'] >= 4)]
  if user_ratings.empty:
    print(f"No high-rated movies found for user {user_id}.")
    return
  
  user_name = user_ratings['User_Names'].iloc[0]
  
  # Build a list of genres from the user's high-rated movies (weighted ratings)
  user_genre_weight = Counter()
  for _, row in user_ratings.iterrows():
    weight = row['Rating']
    for genre in row['Genre'].split(','):
      genre = genre.strip()
      if_id_weight = weight / genre_global_frequency[genre]
      user_genre_weight[genre] += if_id_weight
      
  movie_scores = []
  rated_movie_ids = set(user_ratings['Movie_ID'])
  unique_movies = movies_df.drop_duplicates(subset=['Movie_ID'])
  for _, row in unique_movies.iterrows():
    if row['User_ID'] in rated_movie_ids:
      continue # skip rated movies
    score = 0
    for genre in row['Genre'].split(','):
      score += user_genre_weight[genre.strip()]
    movie_scores.append((row['Movie_Title'], row['Genre'], score))
    
  rec_df = pd.DataFrame(movie_scores, columns=['Movie_Title', 'Genre', 'Score'])
  rec_df = rec_df.sort_values(by='Score', ascending=False)
  
  genre_counts = defaultdict(int)
  fina_recommendations = []
  for _, row in rec_df.iterrows():
    movie_genre = row['Genre'].split(',')[0].strip()
    if genre_counts[movie_genre] < 2:
      fina_recommendations.append(row)
      genre_counts[movie_genre] += 1
    if len(fina_recommendations) == top_n:
      break
  
  # Convert list to DataFrame
  final_df = pd.DataFrame(fina_recommendations)

  # Normalize Score column to percentage
  max_score = final_df['Score'].max()
  final_df['Score (%)'] = (final_df['Score'] / max_score) * 100
  final_df['Score (%)'] = final_df['Score (%)'].round(2)

  # Drop raw score if not needed
  final_df = final_df.drop(columns=['Score'])
  print("Running CONTENT BASED RECOMMENDATION SYSTEM")
  print(f"\nTop 10 recommendations for {user_name} (User ID: {user_id}):")
  print("-----------------------------------------")
  return final_df
  
recommendations = content_based_recommendation(user_id=501, movies_df=movies_df)
print(recommendations)