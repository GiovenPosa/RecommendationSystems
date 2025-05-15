# 🎬 Movie Recommendation Engine

A multi-model movie recommendation system built in Python using **SVD matrix factorization**, user-based **collaborative filtering**, and **content-based filtering**. Developed as part of my self-directed learning in applied AI and data science, this project explores the strengths and limitations of different recommendation approaches.

The engine is designed with modularity in mind to support future expansion into **hybrid recommendation** systems that intelligently combine collaborative and content-based signals. As part of this ongoing development, I aim to compare the **robustness**, **accuracy**, and **cold-start handling** of each method, evaluating their performance in different data sparsity and user interaction scenarios.

---

## 🧠 Project Purpose

This project was created to:
- Deepen my understanding of **collaborative filtering** and **conten-based filtering** techniques in AI.
- Explore both **model-based** and **memory-based** recommender systems.
- Analyse the effectiveness of **hybrid engines** by combining these two approaches.
- Build a working prototype that mirrors real-world recommendation engines.
- Demonstrate my growing skills in **machine learning, Python**, and **data analysis**.
- Extend my Master's-level academic learning with practical application.

---

## 🚀 What It Does

This engine predicts and recommends movies to a target user based on prior user ratings using two different techniques:

### 1. **User-User Collaborative Filtering (Memory-Based)**
- Uses `scikit-learn` to compute cosine similarity between users.
- Predicts ratings based on ratings from similar users.
- No model training required; recommendations are computed on the fly.

### 2. **SVD-Based Collaborative Filtering (Model-Based)**
- Uses matrix factorization from the `surprise` library.
- Learns latent features to estimate how a user would rate unseen movies.
- Returns top-N highest-rated predictions for a given user.

### 3.	Content-Based Filtering (Memory-Based)
- Builds a user profile based on genres of movies they’ve rated highly (≥ 4).
- Uses rating-weighted genre frequency to personalize recommendations.
- Scores and ranks unseen movies by comparing genre overlap with the user’s preferences.
- Normalizes and returns top-N movies aligned with the user’s taste.

---

## 🎯 Why I Built This

As a Master’s student in Computer Science preparing for a career in AI-powered applications and software engineering, this project helped me:

	•	Understand how recommender systems work under the hood.
	•	Gain practical experience using machine learning libraries.
	•	Learn how to balance model interpretability vs. performance.
	•	Practice working with real datasets and noisy data.
	•	Develop software that could be integrated into larger systems, such as media apps or e-commerce platforms.

 ### 🧠 Key Concepts Explored

1. Matrix Factorization (SVD)
2. Cross Validation Techniques
3. Collaborative Filtering
5. Cosine Similarity
6. Cold Start Problem
7. Evaluation Techniques
8. Data Sparsity Handling
9. Content-Based Filtering
10. User-Profiling

---

## ⚒️ Development Process

This project was built iteratively, with the following key phases:
### 1. Data Cleaning and Preperation
I merged user ratings with movie metdadata using pandas and normalised inputs for modelling. 

### 2. User-User Collaborative Filtering with Cosin Similarity
Built a manual implementation using `sklearn.metrics.pairwise.cosing_similarity` on ratings matrix. I found this good but I wanted to test how good the results are by comparing it to a different technique model.

#### Sparsity Check
After checking for sparsity of the user dataset, I have learned that the current dataset *Extended_Movie_data.csv* scores very highly (>95%) in the sparsity of **97%**, which means that most users have not rated most movies - not ideal for user-based CF as it relies on rating overlap (since I'm currently limited to ratings for user data). This is my code to check for ratings sparsity: 
```
ratings_matrix = movies_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
sparsity = 1.0 - (ratings_matrix.count().sum() / (ratings_matrix.shape[0] * ratings_matrix.shape[1]))
print(f"Matrix Sparsity: {sparsity:.2%}")
```

### 3. SVD Integration
I decided that I wanted to use a matrix factorization technique used in machine learning for dimensionality reduction, noise reduction, and feature extraction. I used the surpirce library to fit a MF model using the same dataset. 

### 4. Testing and Comparison
I wrote utility functions to outpur and compare top recommendations between both approaches for the same user at multiple iterations. I was not confident on how relaible the results are with SVD so i explored further on how **seed** can affect results.

### 5. Exploration of Random Seeds and Results Variance
After doing some research, I was introducde to a method of analysing RSME and MAE. I randomly loaded in 20 - 50 unique seed states into an array and initialised the SVD model in a loop, iterating and returning the result of each unique seed.
Lowest seeds tend to suggest a better-performing model - it predicts ratings most accurately (but i am not sure how reliable this theory is). 

Each time I run the loop, a different seed and RSME list is returned. This is where I pivoted from using `train_test_split` and explored the use of `cross_validation` at `cv=5` to evaluate the model multiple times on different splits of the dataset (average the results). 

| Problems with Single-Split | Solution via Cross-Validation |
|-----------------|-------------|
| Performance may depend on which data got into the test set | Multiple test sets are used |
| A lucky or unlucky test split might be misleading | Averages smooth out randomness |
| Doesnt show vairability in results | Noticeable stability of the model during testing |
| Might overfit to that specific test set | Less liekly to overfit on multiple folds |

*Note: Some considerations I need to test include: testing with different fold metrics; compare standard deviation of RSMEs across folds, use other metrics for recommendation like 'time-stamp', 'genre', etc; and improve the current dataset.*

### 6. First Steps Into Hybrid Engine (Memory Based) 

 
