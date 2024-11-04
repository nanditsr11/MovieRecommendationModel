## Movie Recommendation System

# Overview
This project implements a simple movie recommendation system using collaborative filtering based on user ratings. It leverages user similarity calculated using cosine similarity to predict missing ratings and recommend movies to users. The system allows users to input their ID and receive personalized movie recommendations based on ratings from similar users.

## Features
User Rating Predictions: Predicts missing ratings for movies using collaborative filtering techniques.
Movie Recommendations: Provides personalized movie recommendations based on user similarity.
User-Friendly Interface: Simple command-line interface for user interaction.
CSV Output: Saves the updated ratings data to a CSV file after execution.

## Requirements
Python 3.x
Pandas
NumPy
scikit-learn

## Installation
To set up the project, follow these steps:

1. Clone the repository (or download the script):

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

2. Install the required packages: You can install the required Python packages using pip:

pip install pandas numpy scikit-learn

3. Run the script: Execute the script in your terminal:

python movie_recommendation.py

## Usage
Initial Ratings: The system starts with a predefined dataset of user ratings, including some missing values.
Predict Ratings: The script automatically predicts missing ratings based on user similarity.
Get Recommendations: Enter a valid user ID when prompted to receive movie recommendations. Valid IDs are displayed for user convenience.
Exit: Type 0 to exit the program.

## Output
The script saves the updated ratings DataFrame to a CSV file named final_ratings.csv in the movie_recommendations directory after completion.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Inspired by collaborative filtering techniques in recommendation systems.
Thanks to the contributors of the libraries used in this project.
