import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt

# Load CSV data to DataFrame
column_names = ['User_ID', 'User_Name', 'Movie_ID', 'Rating']
ratings_df = pd.read_csv('Extended_Movie_data.csv', sep=',', names=column_names)

# Load titles to map Movie_ID to names
titles_df = pd.read_csv("Extended_Movie_Id_Titles.csv")
titles_df.rename(columns={'item_id': 'Movie_ID', 'title': 'Movie_Title'}, inplace=True)

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['User_ID', 'Movie_ID', 'Rating']], reader)

# Split the data into training and testing sets
# trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

seeds = [3, 7, 10, 12, 18, 21, 29, 34, 38, 42, 46, 61, 63, 65, 70, 77, 79, 81, 87, 93]

## **COMEBACK TO THIS LATER**
## use this to find the top 20 seeds out of 100 iterations to find the best seed
""" for i in range(20):
  seeds.append(random.randint(0, 100)) """
  
results = []

for seed in seeds:
  model = SVD(random_state=seed)
  cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
  
  avg_rmse = round(np.mean(cv_results['test_rmse']), 4)
  avg_mae = round(np.mean(cv_results['test_mae']), 4)
  results.append({'Seed': seed, 'RMSE': avg_rmse, 'MAE': avg_mae})
  
results_df = pd.DataFrame(results).sort_values('RMSE')

# If you want to output the graph RSME results UNCOMMENT THIS part!
""" plt.figure(figsize=(10, 6))
plt.scatter(results_df['Seed'], results_df['RMSE'], marker='o', label='RMSE', color='green')
plt.title('RMSE for Different Seeds')
plt.xlabel('Random Seed')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("RMSE and MAE for different seeds:")
print(results_df) """

# Fix the best-performing seed
best_seed = results_df.iloc[0]['Seed']
model = SVD(random_state=int(best_seed))

# Train the model on the entire dataset
trainset = data.build_full_trainset()
model.fit((trainset))

def recommend_svd(user_id, titles_df, model, ratings_df, top_n=10, seed=best_seed):
  
  # Get all movie IDs
  all_movie_ids = titles_df['Movie_ID'].unique()
  user_name = ratings_df[ratings_df['User_ID'] == user_id]['User_Name'].values[0]
  
  # Filter out movies already reated by the user
  rated_movies = ratings_df[ratings_df['User_ID'] == user_id]['Movie_ID'].unique()
  candidate_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
  
  # Predict ratings for all unrated movies 
  predictions = [model.predict(user_id, movie_id) for movie_id in candidate_movies]
  predictions.sort(key=lambda x: x.est, reverse=True)
  
  # Show top N results
  top_predictions = predictions[:top_n]
  
  print("Running SVD RECOMMENDATION SYSTEM")
  print(f"\nSeed Used: {best_seed}")
  print(f"\nTop {top_n} recommendations for User {user_name} ({user_id}): ")  
  print("------------------------------------------------------------")
  for pred in top_predictions:
    title = titles_df[titles_df['Movie_ID'] == pred.iid]['Movie_Title'].values[0]
    percentage_scores = pred.est * 20  # Convert to percentage
    print(f"{title} (Movie ID: {pred.iid}) -- {percentage_scores:.2f}% You Might Like It")
    
# Run SVD recommendation engine here!!!
recommend_svd(user_id=501, titles_df=titles_df, model=model, ratings_df=ratings_df, top_n=10, seed=best_seed)

