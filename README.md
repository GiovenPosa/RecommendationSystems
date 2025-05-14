# 🎬 Movie Recommendation Engine

A dual-model recommendation system built in Python using **SVD matrix factorization** and **user-based collaborative filtering**, developed as part of my self-directed learning in applied AI and data science.

---

## 🧠 Project Purpose

This project was created to:
- Deepen my understanding of **collaborative filtering** techniques in AI.
- Explore both **model-based** and **memory-based** recommender systems.
- Build a working prototype that mirrors real-world recommendation engines.
- Demonstrate my growing skills in **machine learning, Python**, and **data analysis**.
- Extend my Master's-level academic learning with practical application.

---

## 🚀 What It Does

This engine predicts and recommends movies to a target user based on prior user ratings using two different techniques:

### 1. **SVD-Based Collaborative Filtering (Model-Based)**
- Uses matrix factorization from the `surprise` library.
- Learns latent features to estimate how a user would rate unseen movies.
- Returns top-N highest-rated predictions for a given user.

### 2. **User-User Collaborative Filtering (Memory-Based)**
- Uses `scikit-learn` to compute cosine similarity between users.
- Predicts ratings based on ratings from similar users.
- No model training required; recommendations are computed on the fly.

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
2. Collaborative Filtering
3. Cosine Similarity
4. Cold Start Problem
5. Evaluation Techniques
6. Data Sparsity Handling

---

## ⚒️ Development Process

This project was built iteratively, with the following key phases:
### 1. Data Cleaning and Preperation
I merged user ratings with movie metdadata using pandas and normalised inputs for modelling. 

### 2. User-User Collaborative Filtering with Cosin Similarity
Built a manual implementation using `sklearn.metrics.pairwise.cosing_similarity` on ratings matrix. I found this good but I wanted to test how good the results are by comparing it to a different technique model.

### 3. SVD Integration
I decided that I wanted to use a matrix factorization technique used in machine learning for dimensionality reduction, noise reduction, and feature extraction. I used the surpirce library to fit a MF model using the same dataset.

### 4. Testing and Comparison
I wrote utility functions to outpur and compare top recommendations between both approaches for the same user at multiple iterations. I was not confident on how relaible the results are with SVD so i explored further on how **seed** can affect results.

### 5. Exploration of Random Seeds and Results Variance
After doing some research, I was introducde to a method of analysing RSME and MAE. I randomly loaded in 20 - 50 unique seed states into an array and initialised the SVD model in a loop, iterating and returning the result of each unique seed.
Lowest seeds tend to suggest a better-performing model - it predicts ratings most accurately (but i am not sure how reliable this theory is). 

Each time I run the loop, a different seed and RSME list is returned. 

 
