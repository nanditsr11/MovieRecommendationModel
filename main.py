import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Step 1: Create a simple dataset of users, movies, and their ratings with some missing ratings
data = {
    'UserID': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'MovieID': [101, 102, 103, 101, 104, 102, 103, 104, 101, 103],
    'Rating': [5, 4, np.nan, 5, 2, 4, np.nan, 5, 3, 4]  # Introducing missing ratings
}

# Create a DataFrame
ratings_df = pd.DataFrame(data)
print("Initial Ratings DataFrame with Missing Ratings:")
print(ratings_df)

# Step 2: Create a user-item matrix
user_movie_matrix = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')

# Compute cosine similarity between users, ignoring NaN values
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Step 3: Define a function to recommend movies based on user similarity
def recommend_movies(user_id, user_movie_matrix, user_similarity_df, n_recommendations=3):
    # Check if the user_id is valid
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"User ID {user_id} is out of range. Valid User IDs are: {user_movie_matrix.index.tolist()}")

    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Get the movies rated by the similar users
    similar_users_ratings = user_movie_matrix.loc[similar_users.index]

    # Calculate weighted ratings
    weighted_ratings = similar_users_ratings.multiply(similar_users, axis=0).sum(axis=0)

    # Normalize weighted ratings by the sum of similarities
    recommendation_scores = weighted_ratings / similar_users.sum()
    
    # Remove already rated movies
    already_rated = user_movie_matrix.loc[user_id]
    recommendation_scores = recommendation_scores[already_rated[already_rated.notna()].index]
    
    # Get the top N movie recommendations
    recommended_movies = recommendation_scores.sort_values(ascending=False).head(n_recommendations)
    
    return recommended_movies.index.tolist()

# Function to predict missing ratings
def predict_ratings(user_movie_matrix, user_similarity_df):
    for user_id in user_movie_matrix.index:
        for movie_id in user_movie_matrix.columns:
            if pd.isna(user_movie_matrix.at[user_id, movie_id]):
                # Get similar users
                similar_users = user_similarity_df[user_id].sort_values(ascending=False)
                
                # Get ratings from similar users for this movie
                similar_users_ratings = user_movie_matrix.loc[similar_users.index, movie_id]
                
                # Calculate the mean rating of similar users who rated this movie
                predicted_rating = similar_users_ratings.mean()
                
                # Only assign the predicted rating if it is not NaN
                if not pd.isna(predicted_rating):
                    user_movie_matrix.at[user_id, movie_id] = predicted_rating

# Predict missing ratings
predict_ratings(user_movie_matrix, user_similarity_df)

# Display updated ratings matrix
print("\nUser-Movie Matrix after Predicting Missing Ratings:")
print(user_movie_matrix)

# Main loop to get user input for recommendations
while True:
    try:
        # Prompt for user input
        user_id = int(input(f"Enter User ID (valid IDs are: {list(user_movie_matrix.index)} or 0 to exit): "))
        
        if user_id == 0:
            print("Exiting the program.")
            break
        
        recommendations = recommend_movies(user_id, user_movie_matrix, user_similarity_df)
        print(f"\nRecommended movies for User {user_id}: {recommendations}")

    except ValueError as e:
        print(e)
        print("Please enter a valid User ID.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Save the updated ratings DataFrame to a CSV file after the whole program is run
output_directory = "movie_recommendations"
os.makedirs(output_directory, exist_ok=True)
ratings_df.to_csv(os.path.join(output_directory, "final_ratings.csv"), index=False)
print("\nAll ratings saved to final_ratings.csv.")
