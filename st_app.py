import streamlit as st
import pickle
import pandas as pd
import requests


def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US')
    data = response.json()
    return "https://image.tmdb.org/t/p/w185/" +data['poster_path']

# Load pickled data
with open("recommendation_system_data.pickle", "rb") as f:
    pickle_data = pickle.load(f)

tfidf = pickle_data["tfidf"]
cosine_sim = pickle_data["cosine_sim"]
cosine_sim2 = pickle_data["cosine_sim2"]
indices = pickle_data["indices"]
movies_df = pickle_data["movies_df"]
movies = pd.read_csv("tmdb_5000_credits.csv")
movies_list = movies_df['title'].values
print(movies.head())

C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)

# Function to get content-based recommendations
def get_content_based_recommendations(title, cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first entry (itself), and select top 10 similar movies
    recommended_movies = []
    recommend_movie_poster = []
    for i in sim_scores:
        movie_id = movies.iloc[i[0]].movie_id
        print(movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from api
        recommend_movie_poster.append(fetch_poster(movie_id))
    # print(recommended_movies)
    return recommended_movies,recommend_movie_poster

# Function to get metadata-based recommendations
def get_metadata_based_recommendations(title, cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first entry (itself), and select top 10 similar movies
    # movie_indices = [ind[0] for ind in sim_scores]
    recommended_movies = []
    recommend_movie_poster = []
    for i in sim_scores:
        movie_id = movies.iloc[i[0]].movie_id
        print(movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from api
        recommend_movie_poster.append(fetch_poster(movie_id))
    # print(recommended_movies)
    return recommended_movies,recommend_movie_poster

def get_demographic_recommendations(new_movies_df, C, m):
    new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
    new_movies_df = new_movies_df.sort_values('score', ascending=False)
    recommendations = new_movies_df["title"].head(10).to_list()
    recommend_movie_poster = []
    for i in recommendations:
        movie_id = new_movies_df[new_movies_df["title"] == i]["movie_id"].iloc[0]
        # fetch poster from api
        recommend_movie_poster.append(fetch_poster(movie_id))
    return recommendations ,recommend_movie_poster

# Function to get collaborative recommendations
def get_collaborative_recommendations():
    return "Under construction..."

# Main Streamlit app
st.title('Movie Recommendation System')

option = st.sidebar.selectbox('Select Recommendation Type', ['Content-Based', 'Demographic', 'Collaborative'])

if option == 'Content-Based':
    st.subheader('Content-Based and Metadata Recommendations')

    selected_movie = st.selectbox('Select a movie:', movies_df['title'].values, key='content_based_selectbox', index=None)

    if st.button('Get Recommendations'):
        
        if selected_movie !=None:
            st.write('Title : ', selected_movie)
            st.write(f"Genres: {', '.join(movies_df.loc[movies_df['title'] == selected_movie, 'genres'].values[0])}")

            st.write("Director:", movies_df.loc[movies_df['title'] == selected_movie, 'director'].values[0])
            st.write("Cast:", ' '.join(movies_df.loc[movies_df['title'] == selected_movie, 'cast'].values[0]))
            
            names_c,poster_c = get_content_based_recommendations(selected_movie, cosine_sim)
            names,poster = get_metadata_based_recommendations(selected_movie, cosine_sim2)

            # Display recommendations side by side

            with st.container():
                st.write(f"Recommendations for {selected_movie} based on content:")
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    st.text(names_c[0])
                    st.image(poster_c[0])
                with col2:
                    st.text(names_c[1])
                    st.image(poster_c[1])
                with col3:
                    st.text(names_c[2])
                    st.image(poster_c[2])
                with col4:
                    st.text(names_c[3])
                    st.image(poster_c[3])
                with col5:
                    st.text(names_c[4])
                    st.image(poster_c[4])
            with st.container():
                st.write(f"Recommendations for {selected_movie} based on metadata:")
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    st.text(names[0])
                    st.image(poster[0])
                with col2:
                    st.text(names[1])
                    st.image(poster[1])
                with col3:
                    st.text(names[2])
                    st.image(poster[2])
                with col4:
                    st.text(names[3])
                    st.image(poster[3])
                with col5:
                    st.text(names[4])
                    st.image(poster[4])
                
                # st.write(metadata_recommendations)
        else:
            st.error("Please Select a Movie")

elif option == 'Demographic':
    st.subheader('Demographic Filtering Recommendations')
    new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
    # recommendations = get_demographic_recommendations(new_movies_df, C, m)
    # st.write("Top 10 Popular and Well-Rated Movies :")
    # st.write(recommendations)
    names_d,poster_d = get_demographic_recommendations(new_movies_df, C, m)
    with st.container():
        st.write("Top 10 Popular and Well-Rated Movies :")
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.text(names_d[0])
            st.image(poster_d[0])
        with col2:
            st.text(names_d[1])
            st.image(poster_d[1])
        with col3:
            st.text(names_d[2])
            st.image(poster_d[2])
        with col4:
            st.text(names_d[3])
            st.image(poster_d[3])
        with col5:
            st.text(names_d[4])
            st.image(poster_d[4])
            
    with st.container():
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.text(names_d[5])
            st.image(poster_d[5])
        with col2:
            st.text(names_d[6])
            st.image(poster_d[6])
        with col3:
            st.text(names_d[7])
            st.image(poster_d[7])
        with col4:
            st.text(names_d[8])
            st.image(poster_d[8])
        with col5:
            st.text(names_d[9])
            st.image(poster_d[9])
            
else:
    st.subheader('Collaborative Filtering Recommendations')
    st.write(get_collaborative_recommendations())
