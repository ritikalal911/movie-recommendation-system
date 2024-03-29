# Movie Recommendation System

This is a simple movie recommendation system built using Streamlit, which provides three types of recommendations: Content-Based, Demographic Filtering, and Collaborative Filtering.

## How to Run

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Ensure you have the necessary data files (`tmdb_5000_credits.csv` and `tmdb_5000_movies.csv`) in the project directory.
5. Run the Streamlit app by executing `streamlit run st_app.py`.
6. Access the app in your web browser via the provided URL.

## Features

- **Content-Based Filtering**: Allows users to select a movie and get recommendations based on similarity to the selected movie's overview.
- **Demographic Filtering**: Displays the top popular and well-rated movies.
- **Collaborative Filtering**: Under construction.

## Requirements

- Python 3.x
- Streamlit
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Files

- `main.py`: Contains the main logic for loading data and defining recommendation functions.
- `st_app.py`: Streamlit app for user interaction.
- `recommendation_system_data.pickle`: Pickled data containing TF-IDF matrices and cosine similarity scores.
- `tmdb_5000_credits.csv`: CSV file containing movie credits data.
- `tmdb_5000_movies.csv`: CSV file containing movie metadata.

## Dataset

The dataset used for this project is the TMDB 5000 Movie Dataset, which contains information about movies including titles, genres, cast, crew, and user ratings.

## Credits

- Streamlit: https://streamlit.io/
- TMDB 5000 Movie Dataset: https://www.kaggle.com/tmdb/tmdb-movie-metadata

